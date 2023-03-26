import json
import nltk
import re

lowerThreshold = 0.40   
stopWords = set(nltk.corpus.stopwords.words("english"))      

def sendErrorResponse(statusCode, errMessage):
    return {
        "statusCode": statusCode,
        "body": json.dumps(
            {
                "message": errMessage
            }
        ),
    }

def cleanData(lines):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    finalLines = []

    for line in lines:
        # Remove the punctuation from the lines:
        line = re.sub(r'[^\w\s]', '', line)
        result = ""
        wordList = line.split(" ")
        for word in wordList:
            # Convert string to lower case characters:
            resultantWord = word.lower()
            # Remove stop words:
            if(resultantWord in stopWords):
                resultantWord = ""
            # Perfrom Lemmatization:
            resultantWord = lemmatizer.lemmatize(resultantWord)
            if(len(resultantWord)!=0):
                if(len(result)==0):
                    result = resultantWord
                else:
                    result += " " + resultantWord
        finalLines.append(result)

    return finalLines

def access(i, j, dp):
    if(i>=0 and j>=0):
        return dp[i][j]
    elif(j >=0):
        return j+1
    else:
        return i+1

# 1. One of the possible way of comparing sentences:
def levenshtein_similarity(sentence1, sentence2):
    sentence1 = sentence1.split(" ")
    sentence2 = sentence2.split(" ")
    if(len(sentence1)==0):
        return len(sentence2)
    if(len(sentence2)==0):
        return len(sentence1)
    dp = [[0]*len(sentence2) for _ in range(len(sentence1))]
    for i in range(len(sentence1)):
        for j in range(len(sentence2)):
            if(sentence1[i]==sentence2[j]):
                dp[i][j] = access(i-1, j-1, dp)
            else:
                dp[i][j] = 1 + min(min(access(i-1, j-1, dp), access(i-1, j, dp)), access(i, j-1, dp))
    distance = dp[len(sentence1)-1][len(sentence2)-1]/max(len(sentence1), len(sentence2))
    return 1-distance

def computeSimilarity(srcDoc, candDoc):
    # Suspected instances of plagarism:
    matches = []

    # Sentence tokenise the file data:
    srcDocTokenised = nltk.tokenize.sent_tokenize(srcDoc)
    candDocTokenised = nltk.tokenize.sent_tokenize(candDoc)

    # Word Tokenise each sentence:
    srcDocTokenisedCleaned = cleanData(srcDocTokenised)
    candDocTokenisedCleaned = cleanData(candDocTokenised)

    # Get the similarity for lines in the candidate document from source document:
    globalSimilarity  = 0.0
    for i in range(len(candDocTokenisedCleaned)):
        similarity = 0.0
        expectedMatchIndex = -1
        for j in range(len(srcDocTokenisedCleaned)):
            currentSimilarity = levenshtein_similarity(candDocTokenisedCleaned[i], srcDocTokenisedCleaned[j])
            if(similarity < currentSimilarity):
                similarity = currentSimilarity
                expectedMatchIndex = j
        globalSimilarity += similarity
        if(similarity > lowerThreshold):
            matches.append({"sourceDocument": srcDocTokenised[expectedMatchIndex], "candidateDocument":candDocTokenised[i]})
    
    # Get the average similarity score:
    globalSimilarity = globalSimilarity / len(candDocTokenisedCleaned)

    return globalSimilarity, matches


def lambda_handler(event, context):
    body = json.loads(event['body'])

    # Handle error cases:
    if("srcText" not in body):
        return sendErrorResponse(400, "Missing: srcText field not provided")

    if("candText" not in body):
        return sendErrorResponse(400, "Missing: candText field not provided")
    
    similarity, matches = computeSimilarity(body["srcText"], body["candText"])

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
                "similarity": similarity,
                "matches": matches
            }
        ),
    }