import os
import boto3
import json
import numpy as np
import cv2
import copy
import base64

delimiter = "##EE##"
maxNoOfAllowedChars = 2048
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


def convertASCIIStringToBinaryString(message):
    result = ""
    for ch in message:
        result += "{0:08b}".format(ord(ch))
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

def hideDataToImage(message, secretKey, srcImage):

    # The imread unchanged is necessary to prevent converting single channel to three channel data (by duplication of same value into BGR layers)
    srcImageOriginal = copy.deepcopy(srcImage) # For PSNR calculation later on
    message += delimiter
    binaryMessage = convertASCIIStringToBinaryString(message)

    # Process based on whether the image is three channel (RGB) or single channel (grayscale):
    if(srcImage.ndim==2):
        if(len(binaryMessage)>=srcImage.shape[0]*srcImage.shape[1]):
            print("The message can't be econded as its length is too high")
            return
        x = getPermutedArray(secretKey, srcImage.shape[0])
        y = getPermutedArray(secretKey, srcImage.shape[1])
        count = 0
        for i in x:
            for j in y:
                if(count==len(binaryMessage)):
                    break
                if(binaryMessage[count]=='0'):
                    if(srcImage[i][j]%2==1):
                        srcImage[i][j] -= 1
                else:
                    if(srcImage[i][j]%2==0):
                        srcImage[i][j] += 1

                count = count + 1
    else:
        if(len(binaryMessage)>=srcImage.shape[0]*srcImage.shape[1]*srcImage.shape[2]):
            print("The message can't be econded as its length is too high")
            return
        x = getPermutedArray(secretKey, srcImage.shape[0])
        y = getPermutedArray(secretKey, srcImage.shape[1])
        z = getPermutedArray(secretKey, srcImage.shape[2])
        count = 0   
        for i in x:
            for j in y:
                for k in z:
                    if(count==len(binaryMessage)):
                        break
                    if(binaryMessage[count]=='0'):
                        if(srcImage[i][j][k]%2==1):
                            srcImage[i][j][k] -= 1
                    else:
                        if(srcImage[i][j][k]%2==0):
                            srcImage[i][j][k] += 1

                    count = count + 1  
    
    print("The data is stored successfully with a psnr of ", cv2.PSNR(srcImage, srcImageOriginal))
    return srcImage


def lambda_handler(event, context):
    try :
        body = json.loads(event['body'])

        # Handle error cases:
        if("message" not in body):
            return sendErrorResponse(400, "Missing: message field not provided")
        
        if("secretKey" not in body):
            return sendErrorResponse(400, "Missing: secretKey field not provided")

        if("fileName" not in body):
            return sendErrorResponse(400, "Missing: fileName field not provided")
        
        if(len(body["message"]) > maxNoOfAllowedChars):
            return sendErrorResponse(400, "Message length exceeds 2048 characters")

        if(len(body["secretKey"])==0):
            return sendErrorResponse(400, "Secret Key can't be empty")

        # Get the object from the S3 bucket
        object = s3.Object(bucket_name, body["fileName"])
        image_content = object.get()['Body'].read()
        nparr = np.frombuffer(image_content, np.uint8)
        srcImage = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

        # Get the response and store it to s3 bucket
        responseImage = hideDataToImage(body["message"], body["secretKey"], srcImage)
        responseImage = cv2.imencode('.png', responseImage)[1].tobytes()
        responseFileName = os.path.splitext(body["fileName"])[0] + ".png"
        s3.Bucket(bucket_name).put_object(Key=responseFileName, Body=responseImage)
        responseUrl = f'https://{bucket_name}.s3.amazonaws.com/{responseFileName}'

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
                    "imageWithDataUrl": responseUrl
                }
            ),
        }
    except Exception as e:
        return sendErrorResponse(500, str(e))
