import json
import numpy as np
import boto3
import os
import cv2
import base64
from scipy.fftpack import dct, idct

#Internal Parameters: Don't change without understanding the code as it may cause hazards
H = 1024        # Host image is resized to this internally for embedding
W = 100         # Watermark image is resized to this internally for embedding
N = 8           # This denotes that we are forming 8x8 sub-blocks
fact = 16       # To cope up with np.uint8 of idct
DCT_ROW = 2     # Row where waterMark is stored in 8x8 dct transform of the image
DCT_COL = 2     # Col where waterMark is stored in 8x8 dct transform of the image

s3 = boto3.resource('s3')
bucket_name = "forensic-tools-s3-bucket"

def sendErrorResponse(statusCode, errMessage):
    return {
        "statusCode": statusCode,
        'headers': {
            'Access-Control-Allow-Headers' : 'Content-Type',
            'Access-Control-Allow-Origin' : '*',
            'Access-Control-Allow-Methods' : 'POST,GET,OPTIONS',
            'Content-Type': 'application/json'
        },
        "body": json.dumps(
            {
                "message": errMessage
            }
        ),
    }

def s3_to_cv2(fileName):
    object = s3.Object(bucket_name, fileName)
    image_content = object.get()['Body'].read()
    nparr = np.frombuffer(image_content, np.uint8)
    srcImage = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    return srcImage

def cv2_to_s3Url(image, format, fileName):
    image = cv2.imencode(format, image)[1].tobytes()
    responseFileName = os.path.splitext(fileName)[0] + format
    s3.Bucket(bucket_name).put_object(Key=responseFileName, Body=image)
    responseUrl = f'https://{bucket_name}.s3.amazonaws.com/{responseFileName}'
    return responseUrl

def binariseImageData (image):
    # Watermark is stored in grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (W, W), interpolation=cv2.INTER_CUBIC)     # This is resizing to W dim
    th , waterMarkImageBinary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    return waterMarkImageBinary

# This idea is motivated by RC4 algorithm to generate randomised permuted array from secret key
def getPermutedArray(secretKey, n):
    S = [i for i in range(n)] 
    T = [0 for i in range(n)]
    for i in range(n):
        T[i] += ord(secretKey[i%len(secretKey)])
        T[i] %= n
    j = 0
    for i in range(n):
        j = (j + S[i] + T[i])%n
        # swapping S[i] & S[j]
        temp = S[i]
        S[i] = S[j]
        S[j] = temp
    return S

def embedWaterMarkInHostImage(hostImage, waterMarkImage, secretKey):

    # Convert to YUV format from BGR format (we will store information in Y channel denoting luminance):
    hostOriginalDim = (hostImage.shape[1], hostImage.shape[0])
    hostImage = cv2.resize(hostImage, (H, H), interpolation=cv2.INTER_CUBIC)
    hostImageY, hostImageU, hostImageV = cv2.split(cv2.cvtColor(hostImage, cv2.COLOR_BGR2YUV))

    # Get the data to embed in binary format:
    waterMarkImageBinary = binariseImageData(waterMarkImage)
    waterMarkImageBinary = waterMarkImageBinary.reshape(-1) # Converts to 1D array
    

    lengthofBinaryString = W*W
    # Do the watermarking:
    numBlocksIn1Dim = H // N
    permutedArray = getPermutedArray(secretKey, numBlocksIn1Dim)
    index = 0
    shouldBreak = False
    
    for i in permutedArray:
        for j in permutedArray:
            BLOCK = hostImageY[8*i:8*i+N, 8*j:8*j+N]
            BLOCK = dct(dct(BLOCK, axis=0, norm='ortho'), axis=1, norm='ortho')
            data = BLOCK[2][2]
            if(waterMarkImageBinary[index]==0):
                data += fact
            else:
                data -= fact
            BLOCK[2][2] = data
            BLOCK = idct(idct(BLOCK, axis=0, norm='ortho'), axis=1, norm='ortho')
            hostImageY[8*i:8*i+N, 8*j:8*j+N] = BLOCK
            index += 1
            if(index==lengthofBinaryString):
                shouldBreak = True
                break
        if(shouldBreak):
            break

    # Combine to get watermarkEmbedded image
    hostImage = cv2.merge((hostImageY, hostImageU, hostImageV))
    hostImage = cv2.cvtColor(hostImage, cv2.COLOR_YUV2BGR)
    hostImage = cv2.resize(hostImage, hostOriginalDim, interpolation=cv2.INTER_CUBIC)

    return hostImage


def lambda_handler(event, context):
    try:
        body = json.loads(event['body'])

        # Handle error cases:
        if("hostImageFileName" not in body):
            return sendErrorResponse(400, "Missing: hostImageFileName field not provided")

        if("waterMarkImageFileName" not in body):
            return sendErrorResponse(400, "Missing: waterMarkImageFileName field not provided")
        
        if("secretKey" not in body):
            return sendErrorResponse(400, "Missing: secretKey field not provided")

        if(len(body["secretKey"])==0):
            return sendErrorResponse(400, "Secret Key can't be empty")
        
        hostImage = s3_to_cv2(body["hostImageFileName"])
        waterMarkImage = s3_to_cv2(body["waterMarkImageFileName"])
        secretKey = body["secretKey"]
        
        imageWithWaterMark = embedWaterMarkInHostImage(hostImage, waterMarkImage, secretKey)
        imageWithWaterMarkUrl = cv2_to_s3Url(imageWithWaterMark, '.jpg', body["hostImageFileName"])

        return {
            "statusCode": 200,
            'headers': {
                'Access-Control-Allow-Headers' : 'Content-Type',
                'Access-Control-Allow-Origin' : '*',
                'Access-Control-Allow-Methods' : 'POST,GET,OPTIONS',
                'Content-Type': 'application/json'
            },
            "body": json.dumps(
                {
                    "message": "success",
                    "imageWithWaterMarkUrl": imageWithWaterMarkUrl
                }
            ),
        }
    except Exception as e:
        return sendErrorResponse(500, str(e))
