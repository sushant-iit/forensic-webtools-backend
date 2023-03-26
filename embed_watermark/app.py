import json
import numpy as np
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

def sendErrorResponse(statusCode, errMessage):
    return {
        "statusCode": statusCode,
        "body": json.dumps(
            {
                "message": errMessage
            }
        ),
    }

def base64_to_cv2(base64_string):
    img_data = base64.b64decode(base64_string)
    img_array = np.frombuffer(img_data, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
    return img

def cv2_to_base64(img):
    _, img_data = cv2.imencode('.png', img)
    base64_string = base64.b64encode(img_data).decode('utf-8')
    return base64_string

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

def embedWaterMarkInHostImage(hostImageStr, waterMarkImageStr, secretKey):

    hostImage = base64_to_cv2(hostImageStr)
    waterMarkImage = base64_to_cv2(waterMarkImageStr)

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

    return cv2_to_base64(hostImage)


def lambda_handler(event, context):
    body = json.loads(event['body'])

    # Handle error cases:
    if("hostImageStr" not in body):
        return sendErrorResponse(400, "Missing: hostImageStr field not provided")

    if("waterMarkImageStr" not in body):
        return sendErrorResponse(400, "Missing: waterMarkImageStr field not provided")
    
    if("secretKey" not in body):
        return sendErrorResponse(400, "Missing: secretKey field not provided")

    if(len(body["secretKey"])==0):
        return sendErrorResponse(400, "Secret Key can't be empty")
    
    imageWithWaterMark = embedWaterMarkInHostImage(body["hostImageStr"], body["waterMarkImageStr"], body["secretKey"])

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
                "imageWithData": imageWithWaterMark
            }
        ),
    }