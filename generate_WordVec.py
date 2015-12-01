from gensim.models import word2vec
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

t0 = time()
model = word2vec.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
t1 = time()
print 'word2vec Read in '+ str(t1-t0) + ' seconds'
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
    print "Total words present =" + str(len(wordName))
    print "Total words missed =" + str(len(wordsNotPresent))
    print "Word vector dimension = "+str(len(wordVec[0]))
    #output.to_csv("wordVectors.csv")
    pass
if __name__ == '__main__':
    #extractSentences()
    getIndependentWordsVector()

    
    