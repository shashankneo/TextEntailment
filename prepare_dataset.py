from gensim.models import word2vec
import pandas as pd
import nltk
import numpy as np
import string
import math
import re
wordList = {}
stopwords = []
wordName= []
wordVec = []
truth = 0
threshold = 0.3
total  = 0

#extracts respective sentence pairs       
OUTPUT_FILE_NAME = "best_data.csv"
INPUT_FILE_NAME = "initial_data.csv"
sentences1 = []
sentences2 = []
entailScore = []
sentence1Column = "QUESTION"
sentence2Column = "ANSWER_SENTENCE"
entailScoreColumn = "ENT_SCORE"
def getSentencePair(input):
     entailScr = input[entailScoreColumn]
     if math.isnan(entailScr) or entailScr == None:
         return
     sent1 = input[sentence1Column]
     sent2 = input[sentence2Column]
     for obj1 in sent1.split("|"):
         for obj2 in sent2.split("|"):
             sentences1.append(obj1)
             sentences2.append(obj2)
             entailScore.append(entailScr)
#Extracts columns for our concern and writes to its respective csv file         
def extractSentences():
    input_data = pd.read_csv(INPUT_FILE_NAME)
    input_data.apply(getSentencePair, axis=1)
    columns = {"sentence1":sentences1 ,"sentence2":sentences2, "entailScore":entailScore}
    output = pd.DataFrame(columns)
    output.to_csv(OUTPUT_FILE_NAME) 
    
def concatInputFiles():
     undergoer_data = pd.read_csv("undergoer.csv")
     undergoer_data = undergoer_data.drop(undergoer_data.columns[[0]], axis=1) 
     
     enabler_data = pd.read_csv("enabler.csv")
     enabler_data = enabler_data.drop(enabler_data.columns[[0]], axis=1)
     
     result_data = pd.read_csv("result.csv")
     result_data = result_data.drop(result_data.columns[[0]], axis=1) 
     
     trigger_data = pd.read_csv("trigger.csv")
     trigger_data = trigger_data.drop(trigger_data.columns[[0]], axis=1)
     
     best_data = pd.read_csv("best_data.csv")
     best_data = best_data.drop(best_data.columns[[0]], axis=1) 

     verticalStack = pd.concat([best_data,undergoer_data, enabler_data,result_data,trigger_data ], axis=0)
     verticalStack.to_csv("inputSentences.csv")
if __name__ == '__main__':
    #extractSentences()
    concatInputFiles()

    
    