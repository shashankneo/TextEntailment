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
    best_data = pd.read_csv("qa_data.csv")
    #best_data = best_data.drop(best_data.columns[[0]], axis=1) 
    
    results = pd.read_csv("/Users/neocfc/Documents/workspace/CompLing/RNN/newresult/prediction/deep2RNNphys_wordvec200/700/cosineSimilarity.csv")
    results['QUESTION'] = best_data['QUESTION']
    results = results.drop(results.columns[[0]], axis=1)
    final_output = results.groupby('QUESTION').apply(lambda t: t[t.avgOurScore==t.avgOurScore.max()])
    final_output.apply(calculatePrecision, axis=1)
    print "Number of questions = " + str(count)
    print "We got right = " + str(ourPrecision)
    print "They got right = " + str(theirPrecision)
if __name__ == '__main__':
    getResult()