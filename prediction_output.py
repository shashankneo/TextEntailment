import pandas as pd

ourPrecision = 0
theirPrecision = 0
count = 0
def calculatePrecision(inputs):
    global count
    global ourPrecision
    global theirPrecision
    question = inputs['QUESTION']
    our_answer = inputs['ANSWER_CHOICE']
    correct_answer  = inputs['CORRECT_ANSWER'] 
    predicted_answer  = inputs['PREDICTED_ANSWER']
    count = count + 1
    if our_answer == correct_answer:
        ourPrecision = ourPrecision + 1
    if predicted_answer == correct_answer:
        theirPrecision = theirPrecision + 1 

def getResult():
    results = pd.read_csv("/Users/neocfc/Documents/workspace/CompLing/RNN/newresult/prediction/biRNN/50000/cosineSimilarity.csv")
    results = results.drop(results.columns[[0]], axis=1)
    final_output = results.groupby('QUESTION').apply(lambda t: t[t.avgOurScore==t.avgOurScore.max()])
    final_output.apply(calculatePrecision, axis=1)
    print "Number of questions = " + str(count)
    print "We got right = " + str(ourPrecision)
    print "They got right = " + str(theirPrecision)
if __name__ == '__main__':
    getResult()