import json
import numpy as np
import cv2
import base64

delimiter = "##EE##"
maxNoOfAllowedChars = 2048

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

def convertASCIIStringToBinaryString(message):
    result = ""
    for ch in message:
        result += "{0:08b}".format(ord(ch))
    return result

def convertBinaryStringToASCII(binaryString):
    out = [binaryString[i:i+8] for i in range(0, len(binaryString), 8)]
    asciiVal = [int(x, 2) for x in out]
    result = ""
    for x in asciiVal:
        result += chr(x)
    return result

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

def retrieveDataFromImage(secretKey, srcImageString):
    # The imread unchanged is necessary to prevent converting single channel to three channel data (by duplication of same value into BGR layers)
    srcImage = base64_to_cv2(srcImageString)
    binaryMessage = ""
    binaryDelimiter = convertASCIIStringToBinaryString(delimiter)
    binaryDelimiterSize = len(binaryDelimiter)
    shouldBreak = False
    errorStatus = False
    # Process based on whether the image is three channel (RGB) or single channel (grayscale):
    if(srcImage.ndim==2):
        x = getPermutedArray(secretKey, srcImage.shape[0])
        y = getPermutedArray(secretKey, srcImage.shape[1])
        for i in x:
            for j in y:
                if(srcImage[i][j]%2==0):
                    binaryMessage += "0"
                else:
                    binaryMessage += "1"
                if(len(binaryMessage)>=binaryDelimiterSize and binaryMessage[len(binaryMessage)-binaryDelimiterSize:]==binaryDelimiter):
                    shouldBreak = True
                    break
                if(len(binaryMessage)-binaryDelimiterSize > maxNoOfAllowedChars*8):
                    errorStatus = True
                    shouldBreak = True
                    break
            if(shouldBreak):
                break
    else:
        x = getPermutedArray(secretKey, srcImage.shape[0])
        y = getPermutedArray(secretKey, srcImage.shape[1])
        z = getPermutedArray(secretKey, srcImage.shape[2])
        for i in x:
            for j in y:
                for k in z:
                    if(srcImage[i][j][k]%2==0):
                        binaryMessage += "0"
                    else:
                        binaryMessage += "1"
                    if(len(binaryMessage)>=binaryDelimiterSize and binaryMessage[len(binaryMessage)-binaryDelimiterSize:]==binaryDelimiter):
                        shouldBreak = True
                        break 
                    if(len(binaryMessage)-binaryDelimiterSize > maxNoOfAllowedChars*8):
                        errorStatus = True
                        shouldBreak = True
                        break
                if(shouldBreak):
                    break
            if(shouldBreak):
                break
    # Remove the delimiter:
    binaryMessage = binaryMessage[:len(binaryMessage)-binaryDelimiterSize]
    finalDecodeMessage = convertBinaryStringToASCII(binaryMessage)
    return errorStatus, finalDecodeMessage
    
def lambda_handler(event, context):
    body = json.loads(event['body'])

    # Handle error cases:
    if("secretKey" not in body):
        return sendErrorResponse(400, "Missing: secretKey field not provided")

    if("imageString" not in body):
        return sendErrorResponse(400, "Missing: imageString field not provided")

    errorStatus, decodedMessage = retrieveDataFromImage(body["secretKey"], body["imageString"])

    if(errorStatus):
        return sendErrorResponse(400, "Either secretKey is wrong or message size exceeds 2048 characters")

    return {
        "statusCode": 200,
        "body": json.dumps(
            {
                "message": "success",
                "retrievedData": decodedMessage
            }
        ),
    }
