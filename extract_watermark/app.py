import json
import numpy as np
import cv2
import base64
from scipy.fftpack import dct

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

def extractWaterMarkImage(imageEmbeddedWithWaterMarkStr, secretKey):

    # Get the data:
    imageEmbeddedWithWaterMark = base64_to_cv2(imageEmbeddedWithWaterMarkStr)
    imageEmbeddedWithWaterMark = cv2.resize(imageEmbeddedWithWaterMark, (H, H), interpolation=cv2.INTER_CUBIC)
    imageEmbeddedWithWaterMarkY, _, _ = cv2.split(cv2.cvtColor(imageEmbeddedWithWaterMark, cv2.COLOR_BGR2YUV))

    waterMarkImageBinary = ""
    index = 0
    shouldBreak = False
    lengthofBinaryString = W*W
    numBlocksIn1Dim = H // N
    permutedArray = getPermutedArray(secretKey, numBlocksIn1Dim)

    # Extract the data:
    for i in permutedArray:
        for j in permutedArray:
            BLOCK = imageEmbeddedWithWaterMarkY[8*i:8*i+N, 8*j:8*j+N]
            # Take DCT:
            BLOCK = dct(dct(BLOCK, axis=0, norm='ortho'), axis=1, norm='ortho')
            data = BLOCK[2][2]
            if(data >= 0):
                waterMarkImageBinary += '0'
            else:
                waterMarkImageBinary += '1'
            index += 1
            if(index==lengthofBinaryString):
                shouldBreak = True
                break
        if(shouldBreak):
            break

   #Save the extracted watermark back:
    waterMarkImageExtracted = [int(waterMarkImageBinary[i])*255 for i in range(0, len(waterMarkImageBinary))]
    waterMarkImageExtracted = np.uint8(waterMarkImageExtracted).reshape((W, W))

    return cv2_to_base64(waterMarkImageExtracted)

def lambda_handler(event, context):
    body = json.loads(event['body'])

    # Handle error cases:
    if("embeddedImageStr" not in body):
        return sendErrorResponse(400, "Missing: embeddedImageStr field not provided")
    
    if("secretKey" not in body):
        return sendErrorResponse(400, "Missing: secretKey field not provided")

    if(len(body["secretKey"])==0):
        return sendErrorResponse(400, "Secret Key can't be empty")
    
    extractedWaterMark = extractWaterMarkImage(body["embeddedImageStr"], body["secretKey"])

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
                "extractedWaterMark": extractedWaterMark
            }
        ),
    }