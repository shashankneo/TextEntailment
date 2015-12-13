from gensim.models import word2vec
import gensim
import pandas as pd
import nltk
import string
import re
from time import time
wordList = {}
stopwords = []
wordName= []
wordVec = []
wordsNotPresent = {}
WORD_VEC_SIZE = 75

model = None

def initStopWordsList():
    global stopwords
    stopwords = []
    f=open("stopwords.txt")
    fileText=f.read("stopwords.txt")
    for word in fileText.split('\n'):
        stopwords.append(word)

def getTokens(filetext):
    no_punctuation = filetext.translate(None, string.punctuation)
    filetext=re.sub("[^a-zA-Z]+", " ", no_punctuation)
    filetext = filetext.lower()
    tokens = nltk.word_tokenize(filetext)
    return tokens

def getIndependentWords(filetext):
    tokens = getTokens(filetext)
    for word in tokens:
        if word not in wordList:
            if word in model:
                wordList[word] = "done"
                wordName.append(word)
                wordVec.append(model[word])
                print word +" = "+model.most_similar(positive = [word])
            else:
                if word not in wordsNotPresent:
                    print word
                    wordsNotPresent[word] = "done"
    pass
    
def getIndependentWordsVector():
   
    inputs = pd.read_csv("inputSentences.csv")
    inputs.drop(inputs.columns[[0]], axis=1) 
    inputs["sentence1"].apply(getIndependentWords)
    inputs["sentence2"].apply(getIndependentWords)
    columns = {"wordName":wordName ,"wordVec":wordVec}
    output = pd.DataFrame(columns)
    output.to_csv("wordVectors_phys_"+str(WORD_VEC_SIZE)+".csv")
    pass

sentenceList = []
def extractSentences(input):
    text = getTokens(input)
    sentenceList.append(text)
def extractSentencesFromPhysical():
    global model
    inputs = pd.read_csv("filtered_sentences.csv")
    inputs["FILTERED_SENTENCE"].apply(extractSentences)
    t0 = time()
    model = gensim.models.Word2Vec(sentenceList, size=WORD_VEC_SIZE, min_count=0)
    t1 = time()
    print 'word2vec Read in '+ str(t1-t0) + ' seconds'
    getIndependentWordsVector()
    print "Total words present =" + str(len(wordName))
    print "Total words missed =" + str(len(wordsNotPresent))
    print "Word vector dimension = "+str(len(wordVec[0]))
if __name__ == '__main__':
    #extractSentences()
    extractSentencesFromPhysical()

    
    