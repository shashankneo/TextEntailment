
import numpy as np
import theano
import theano.tensor as T
import lasagne
import pandas as pd
import nltk
import math
import string
import re
import os
wordList = {}
stopwords = []
wordName= []
wordVec = []


# Min/max sequence length
MIN_LENGTH = 50
MAX_LENGTH = 50
# Number of units in the hidden (recurrent) layer
N_HIDDEN = 50
# Number of training sequences in each batch
N_BATCH = 100
# Optimization learning rate
LEARNING_RATE = .001
# All gradients above this will be clipped
GRAD_CLIP = 100
# How often should we check the output?
EPOCH_SIZE = 100
# Number of epochs to train the net
NUM_EPOCHS = 50
N_FEATURES = 300
n_output = 1
wordVectorDict = {}
train_sentenceVectors1 = []
train_sentenceVectors2 = []
test_undergoerVectors1 = []
test_undergoerVectors2 = []
test_enablerVectors1 = []
test_enablerVectors2 = []
test_triggerVectors1 = []
test_triggerVectors2 = []
test_resultVectors1 = []
test_resultVectors2 = []

gold_cosine_sim_train = []
gold_undergoer_cosine_sim = []
gold_enabler_cosine_sim = []
gold_trigger_cosine_sim = []
gold_result_cosine_sim = []
num_epochs = 100000
def initStopWordsList():
    global stopwords
    stopwords = []
    f=open("stopwords.txt")
    fileText=f.read("stopwords.txt")
    for word in fileText.split('\n'):
        stopwords.append(word)

def getTokens(filetext):
    tokens = " "
    try:
        text = filetext.split("|")
        filetext = " ".join(text)
        no_punctuation = filetext.translate(None, string.punctuation)
        filetext=re.sub("[^a-zA-Z]+", " ", no_punctuation)
        filetext = filetext.lower()
        tokens = nltk.word_tokenize(filetext)
    except Exception,e:
        pass
    return tokens
       
def getTrainSentenceVector1(input):
    zero_word_vec = np.zeros((N_FEATURES,))
    leftTokens = getTokens(input["sentence1"])
    sentVec = []
    leftTokenCount = 0
    for token in leftTokens:
        if token not in wordVectorDict:
            #print token + " in Question"
            pass
        else:
            sentVec.append(wordVectorDict[token]) 
            leftTokenCount = leftTokenCount + 1
    for word in range(leftTokenCount,MAX_LENGTH):
        sentVec.append(zero_word_vec)
    train_sentenceVectors1.append(sentVec)
    
def getTrainSentenceVector2(input):
    zero_word_vec = np.zeros((N_FEATURES,))
    leftTokens = getTokens(input["sentence2"])
    sentVec = []
    leftTokenCount = 0
    for token in leftTokens:
        if token not in wordVectorDict:
            #print token + " in Question"
            pass
        else:
            sentVec.append(wordVectorDict[token]) 
            leftTokenCount = leftTokenCount + 1
    for word in range(leftTokenCount,MAX_LENGTH):
        sentVec.append(zero_word_vec)
    train_sentenceVectors2.append(sentVec)
    
def getTrainCosineSimOutput(input): 
    cosine_sim = input["entailScore"]
    gold_cosine_sim_train.append(cosine_sim)

def getTestUndergoerVector1(input):
    entailScr = input["UNDERGOER_SCORE"]
    zero_word_vec = np.zeros((N_FEATURES,))
    sentVec = []
    leftTokenCount = 0
    if math.isnan(entailScr) or entailScr == None:
        pass
    else:
        leftTokens = getTokens(input["Q_UNDERGOER"])
        for token in leftTokens:
            if leftTokenCount == MAX_LENGTH:
                break
            if token not in wordVectorDict:
                #print token + " in Question"
                pass
            else:
                sentVec.append(wordVectorDict[token]) 
                leftTokenCount = leftTokenCount + 1
    for word in range(leftTokenCount,MAX_LENGTH):
        sentVec.append(zero_word_vec)
    test_undergoerVectors1.append(sentVec)
    
def getTestUndergoerVector2(input):
    entailScr = input["UNDERGOER_SCORE"]
    zero_word_vec = np.zeros((N_FEATURES,))
    sentVec = []
    leftTokenCount = 0
    if math.isnan(entailScr) or entailScr == None:
        pass
    else:
        leftTokens = getTokens(input["A_UNDERGOER"])
        for token in leftTokens:
            if leftTokenCount == MAX_LENGTH:
                break
            if token not in wordVectorDict:
                #print token + " in Question"
                pass
            else:
                sentVec.append(wordVectorDict[token]) 
                leftTokenCount = leftTokenCount + 1
    for word in range(leftTokenCount,MAX_LENGTH):
        sentVec.append(zero_word_vec)
    test_undergoerVectors2.append(sentVec)

def getTestUndergoerCosineSimOutput(input): 
    entailScr = input["UNDERGOER_SCORE"]
    if math.isnan(entailScr) or entailScr == None:
        gold_undergoer_cosine_sim.append(0.0)
    else:
        gold_undergoer_cosine_sim.append(entailScr)
    
def getTestEnablerVector1(input):
    entailScr = input["ENABLER_SCORE"]
    zero_word_vec = np.zeros((N_FEATURES,))
    sentVec = []
    leftTokenCount = 0
    if math.isnan(entailScr) or entailScr == None:
        pass
    else:
        leftTokens = getTokens(input["Q_ENABLER"])
        for token in leftTokens:
            if leftTokenCount == MAX_LENGTH:
                break
            if token not in wordVectorDict:
                #print token + " in Question"
                pass
            else:
                sentVec.append(wordVectorDict[token]) 
                leftTokenCount = leftTokenCount + 1
    for word in range(leftTokenCount,MAX_LENGTH):
        sentVec.append(zero_word_vec)
    test_enablerVectors1.append(sentVec)
    
def getTestEnablerVector2(input):
    entailScr = input["ENABLER_SCORE"]
    zero_word_vec = np.zeros((N_FEATURES,))
    sentVec = []
    leftTokenCount = 0
    if math.isnan(entailScr) or entailScr == None:
        pass
    else:
        leftTokens = getTokens(input["A_ENABLER"])
        for token in leftTokens:
            if leftTokenCount == MAX_LENGTH:
                break
            if token not in wordVectorDict:
                #print token + " in Question"
                pass
            else:
                sentVec.append(wordVectorDict[token]) 
                leftTokenCount = leftTokenCount + 1
    for word in range(leftTokenCount,MAX_LENGTH):
        sentVec.append(zero_word_vec)
    test_enablerVectors2.append(sentVec)

def getTestEnablerCosineSimOutput(input): 
    entailScr = input["ENABLER_SCORE"]
    if math.isnan(entailScr) or entailScr == None:
        gold_enabler_cosine_sim.append(0.0)
    else:
        gold_enabler_cosine_sim.append(entailScr)
        
def getTestTriggerVector1(input):
    entailScr = input["TRIGGER_SCORE"]
    zero_word_vec = np.zeros((N_FEATURES,))
    sentVec = []
    leftTokenCount = 0
    if math.isnan(entailScr) or entailScr == None:
        pass
    else:
        leftTokens = getTokens(input["Q_TRIGGER"])
        for token in leftTokens:
            if leftTokenCount == MAX_LENGTH:
                break
            if token not in wordVectorDict:
                #print token + " in Question"
                pass
            else:
                sentVec.append(wordVectorDict[token]) 
                leftTokenCount = leftTokenCount + 1
    for word in range(leftTokenCount,MAX_LENGTH):
        sentVec.append(zero_word_vec)
    test_triggerVectors1.append(sentVec)
    
def getTestTriggerVector2(input):
    entailScr = input["TRIGGER_SCORE"]
    zero_word_vec = np.zeros((N_FEATURES,))
    sentVec = []
    leftTokenCount = 0
    if math.isnan(entailScr) or entailScr == None:
        pass
    else:
        leftTokens = getTokens(input["A_TRIGGER"])
        for token in leftTokens:
            if leftTokenCount == MAX_LENGTH:
                break
            if token not in wordVectorDict:
                #print token + " in Question"
                pass
            else:
                sentVec.append(wordVectorDict[token]) 
                leftTokenCount = leftTokenCount + 1
    for word in range(leftTokenCount,MAX_LENGTH):
        sentVec.append(zero_word_vec)
    test_triggerVectors2.append(sentVec)

def getTestTriggerCosineSimOutput(input): 
    entailScr = input["TRIGGER_SCORE"]
    if math.isnan(entailScr) or entailScr == None:
        gold_trigger_cosine_sim.append(0.0)
    else:
        gold_trigger_cosine_sim.append(entailScr)

def getTestResultVector1(input):
    entailScr = input["RESULT_SCORE"]
    zero_word_vec = np.zeros((N_FEATURES,))
    sentVec = []
    leftTokenCount = 0
    if math.isnan(entailScr) or entailScr == None:
        pass
    else:
        leftTokens = getTokens(input["Q_RESULT"])
        for token in leftTokens:
            if leftTokenCount == MAX_LENGTH:
                break
            if token not in wordVectorDict:
                #print token + " in Question"
                pass
            else:
                sentVec.append(wordVectorDict[token]) 
                leftTokenCount = leftTokenCount + 1
    for word in range(leftTokenCount,MAX_LENGTH):
        sentVec.append(zero_word_vec)
    test_resultVectors1.append(sentVec)
    
def getTestResultVector2(input):
    entailScr = input["RESULT_SCORE"]
    zero_word_vec = np.zeros((N_FEATURES,))
    sentVec = []
    leftTokenCount = 0
    if math.isnan(entailScr) or entailScr == None:
        pass
    else:
        leftTokens = getTokens(input["A_RESULT"])
        for token in leftTokens:
            if leftTokenCount == MAX_LENGTH:
                break
            if token not in wordVectorDict:
                #print token + " in Question"
                pass
            else:
                sentVec.append(wordVectorDict[token]) 
                leftTokenCount = leftTokenCount + 1
    for word in range(leftTokenCount,MAX_LENGTH):
        sentVec.append(zero_word_vec)
    test_resultVectors2.append(sentVec)

def getTestResultCosineSimOutput(input): 
    entailScr = input["RESULT_SCORE"]
    if math.isnan(entailScr) or entailScr == None:
        gold_result_cosine_sim.append(0.0)
    else:
        gold_result_cosine_sim.append(entailScr)
                
#Form a word vector word2vec representation for each word
def formWordDictFromCsv(input):
    wordName = input["wordName"]
    wordVec = input["wordVec"]
    wordVec = wordVec[1:len(wordVec)-1]
    wordVec= wordVec.replace('\n', '')
    listVec = wordVec.split(" ")
    listVec = filter(None, listVec)
    wordVec_np = np.array(listVec, dtype='str')
    fl_wordVec_np=wordVec_np.astype(np.float)
    wordVectorDict[wordName] = fl_wordVec_np
    pass

#Parses the pair of sentence. Starts the operation to convert sentence into vector of words
def getIndependentWordsVector():
    train_inputs = pd.read_csv("best_data.csv")
    train_inputs = train_inputs.drop(train_inputs.columns[[0]], axis=1)
    wordVec_csv = pd.read_csv("wordVectors.csv")
    wordVec_csv = wordVec_csv.drop(wordVec_csv.columns[[0]], axis=1)
    wordVec_csv.apply(formWordDictFromCsv, axis=1)
    train_inputs.apply(getTrainSentenceVector1, axis=1)
    train_inputs.apply(getTrainSentenceVector2, axis=1)
    train_inputs.apply(getTrainCosineSimOutput, axis=1)
    test_inputs = pd.read_csv("qa_data.csv")
    test_inputs.apply(getTestUndergoerVector1, axis=1)
    test_inputs.apply(getTestUndergoerVector2, axis=1)
    test_inputs.apply(getTestUndergoerCosineSimOutput, axis=1)
    test_inputs.apply(getTestEnablerVector1, axis=1)
    test_inputs.apply(getTestEnablerVector2, axis=1)
    test_inputs.apply(getTestEnablerCosineSimOutput, axis=1)
    test_inputs.apply(getTestTriggerVector1, axis=1)
    test_inputs.apply(getTestTriggerVector2, axis=1)
    test_inputs.apply(getTestTriggerCosineSimOutput, axis=1)
    test_inputs.apply(getTestResultVector1, axis=1)
    test_inputs.apply(getTestResultVector2, axis=1)
    test_inputs.apply(getTestResultCosineSimOutput, axis=1)
    
def gen_csvdata(min_length=MIN_LENGTH, max_length=MAX_LENGTH, n_batch=N_BATCH):
    test_df = pd.read_csv("qa_data.csv")
    getIndependentWordsVector()
    train_len = len(train_sentenceVectors1)
    test_len = len(test_undergoerVectors1)
    #Initialize Mask matrices
    mask_train_1 = np.zeros((train_len, max_length))
    mask_train_2 = np.zeros((train_len, max_length))
    mask_test_undergoer_1 = np.zeros((test_len, max_length))
    mask_test_undergoer_2 = np.zeros((test_len, max_length))
    mask_test_enabler_1 = np.zeros((test_len, max_length))
    mask_test_enabler_2 = np.zeros((test_len, max_length))
    mask_test_trigger_1 = np.zeros((test_len, max_length))
    mask_test_trigger_2 = np.zeros((test_len, max_length))
    mask_test_result_1 = np.zeros((test_len, max_length))
    mask_test_result_2 = np.zeros((test_len, max_length))

    for n in range(train_len):
        sentence_len_1 = len(train_sentenceVectors1[n])
        mask_train_1[n,:sentence_len_1] = 1
        sentence_len_2 = len(train_sentenceVectors1[n])
        mask_train_2[n,:sentence_len_2] = 1
       
    for n in range(test_len):
        sentence_undergoer_len_1 = len(test_undergoerVectors1[n])
        mask_test_undergoer_1[n,:sentence_undergoer_len_1] = 1
        sentence_undergoer_len_2 = len(test_undergoerVectors2[n])
        mask_test_undergoer_2[n,:sentence_undergoer_len_2] = 1       
        
        sentence_enabler_len_1 = len(test_enablerVectors1[n])
        mask_test_enabler_1[n,:sentence_enabler_len_1] = 1
        sentence_enabler_len_2 = len(test_enablerVectors2[n])
        mask_test_enabler_2[n,:sentence_enabler_len_2] = 1 
        
        sentence_trigger_len_1 = len(test_triggerVectors1[n])
        mask_test_trigger_1[n,:sentence_trigger_len_1] = 1
        sentence_trigger_len_2 = len(test_triggerVectors2[n])
        mask_test_trigger_2[n,:sentence_trigger_len_2] = 1  
        
        sentence_result_len_1 = len(test_resultVectors1[n])
        mask_test_result_1[n,:sentence_result_len_1] = 1
        sentence_result_len_2 = len(test_resultVectors2[n])
        mask_test_result_2[n,:sentence_result_len_2] = 1 

    return np.array(train_sentenceVectors1).astype(theano.config.floatX) , np.array(train_sentenceVectors2).astype(theano.config.floatX),\
     np.array(gold_cosine_sim_train).astype(theano.config.floatX),\
     mask_train_1.astype(theano.config.floatX), mask_train_2.astype(theano.config.floatX), \
     np.array(test_undergoerVectors1).astype(theano.config.floatX), np.array(test_undergoerVectors2).astype(theano.config.floatX), \
    np.array(gold_undergoer_cosine_sim).astype(theano.config.floatX),\
     mask_test_undergoer_1.astype(theano.config.floatX), mask_test_undergoer_2.astype(theano.config.floatX),\
    np.array(test_triggerVectors1).astype(theano.config.floatX),  np.array(test_triggerVectors2).astype(theano.config.floatX), \
    np.array(gold_trigger_cosine_sim).astype(theano.config.floatX),\
     mask_test_trigger_1.astype(theano.config.floatX), mask_test_trigger_2.astype(theano.config.floatX),\
     np.array(test_enablerVectors1).astype(theano.config.floatX),  np.array(test_enablerVectors2).astype(theano.config.floatX), \
    np.array(gold_enabler_cosine_sim).astype(theano.config.floatX),\
     mask_test_enabler_1.astype(theano.config.floatX), mask_test_enabler_2.astype(theano.config.floatX),\
     np.array(test_resultVectors1).astype(theano.config.floatX),  np.array(test_resultVectors2).astype(theano.config.floatX), \
    np.array(gold_result_cosine_sim).astype(theano.config.floatX),\
     mask_test_result_1.astype(theano.config.floatX), mask_test_result_2.astype(theano.config.floatX),\
     test_df
    

def averageFinalScore(input):
    count = 0
    total = 0.0
    if input["newUndergoerScore"] != 0 :
        total = total + input["newUndergoerScore"]
        count = count + 1
    if input["newEnablerScore"] != 0 :
        total = total + input["newEnablerScore"]
        count = count + 1
    if input["newTriggerScore"] != 0 :
        total = total + input["newTriggerScore"]
        count = count + 1
    if input["newResultScore"] != 0 :
        total = total + input["newResultScore"]
        count = count + 1
        
    return total/count
def biRNN():
    # First, we build the network, for first sentence starting with an input layer
    # Recurrent layers expect input of shape
    # (batch size, max sequence length, number of features)
    #Giving the batch size as None because we are still experimenting with the 
    # meaning and true usage of the parameter
    #Sequence length corresponds to time steps but this would be variable and it would depend
    #upon the input length so lets give it as None
    #Number of features are 300 because each word is a vector of 300 dimensions
    
   
    l_in_1 = lasagne.layers.InputLayer(shape=(None, MAX_LENGTH, N_FEATURES))
    l_mask_1 = lasagne.layers.InputLayer(shape=(None, MAX_LENGTH))
    
    l_forward_1 = lasagne.layers.RecurrentLayer(
        l_in_1, N_HIDDEN, mask_input=l_mask_1, grad_clipping=GRAD_CLIP,
        W_in_to_hid=lasagne.init.HeUniform(),
        W_hid_to_hid=lasagne.init.HeUniform(),
        nonlinearity=lasagne.nonlinearities.tanh)
    l_forward_deep_1 = lasagne.layers.RecurrentLayer(
        l_forward_1, N_HIDDEN, mask_input=l_mask_1, grad_clipping=GRAD_CLIP,
        W_in_to_hid=lasagne.init.HeUniform(),
        W_hid_to_hid=lasagne.init.HeUniform(),
        nonlinearity=lasagne.nonlinearities.tanh)
    l_backward_1 = lasagne.layers.RecurrentLayer(
        l_in_1, N_HIDDEN, mask_input=l_mask_1, grad_clipping=GRAD_CLIP,
        W_in_to_hid=lasagne.init.HeUniform(),
        W_hid_to_hid=lasagne.init.HeUniform(),
        nonlinearity=lasagne.nonlinearities.tanh,
        backwards=True)
    l_backward_deep_1 = lasagne.layers.RecurrentLayer(
        l_backward_1, N_HIDDEN, mask_input=l_mask_1, grad_clipping=GRAD_CLIP,
        W_in_to_hid=lasagne.init.HeUniform(),
        W_hid_to_hid=lasagne.init.HeUniform(),
        nonlinearity=lasagne.nonlinearities.tanh,
        backwards=True)
    
    l_forward_1_slice = lasagne.layers.SliceLayer(l_forward_deep_1, -1, 1)
    l_backward_1_slice = lasagne.layers.SliceLayer(l_backward_deep_1, 0, 1)
    # Now, we'll concatenate the outputs to combine them.
    l_out_1 = lasagne.layers.ConcatLayer([l_forward_1_slice, l_backward_1_slice])
    # Our output layer is a simple dense connection, with 1 output unit
   
    #l_out_1 = lasagne.layers.SliceLayer(l_concat_1, -1, 1)
    #l_out_1 = lasagne.layers.DenseLayer(l_forward_1, num_units=n_output)
    
    l_in_2 = lasagne.layers.InputLayer(shape=(None, MAX_LENGTH, N_FEATURES))
    l_mask_2 = lasagne.layers.InputLayer(shape=(None, MAX_LENGTH))
    l_forward_2 = lasagne.layers.RecurrentLayer(
        l_in_2, N_HIDDEN, mask_input=l_mask_2, grad_clipping=GRAD_CLIP,
        W_in_to_hid=lasagne.init.HeUniform(),
        W_hid_to_hid=lasagne.init.HeUniform(),
        nonlinearity=lasagne.nonlinearities.tanh)
    l_forward_deep_2 = lasagne.layers.RecurrentLayer(
        l_forward_2, N_HIDDEN, mask_input=l_mask_2, grad_clipping=GRAD_CLIP,
        W_in_to_hid=lasagne.init.HeUniform(),
        W_hid_to_hid=lasagne.init.HeUniform(),
        nonlinearity=lasagne.nonlinearities.tanh)
    
    l_backward_2 = lasagne.layers.RecurrentLayer(
        l_in_2, N_HIDDEN, mask_input=l_mask_2, grad_clipping=GRAD_CLIP,
        W_in_to_hid=lasagne.init.HeUniform(),
        W_hid_to_hid=lasagne.init.HeUniform(),
        nonlinearity=lasagne.nonlinearities.tanh,
        backwards=True)
    l_backward_deep_2 = lasagne.layers.RecurrentLayer(
        l_backward_2, N_HIDDEN, mask_input=l_mask_2, grad_clipping=GRAD_CLIP,
        W_in_to_hid=lasagne.init.HeUniform(),
        W_hid_to_hid=lasagne.init.HeUniform(),
        nonlinearity=lasagne.nonlinearities.tanh,
        backwards=True)
    
    l_forward_2_slice = lasagne.layers.SliceLayer(l_forward_deep_2, -1, 1)
    l_backward_2_slice = lasagne.layers.SliceLayer(l_backward_deep_2, 0, 1)
    # Now, we'll concatenate the outputs to combine them.
    l_out_2 = lasagne.layers.ConcatLayer([l_forward_2_slice, l_backward_2_slice])
    # Our output layer is a simple dense connection, with 1 output unit
   
    #l_out_2 = lasagne.layers.SliceLayer(l_concat_2, -1, 1)
    #l_out_2 = lasagne.layers.DenseLayer(l_forward_2, num_units=n_output)
    
    #target cosine similarity of the pair of sentence
    target_values = T.vector('target_output')
    network_output_1 = lasagne.layers.get_output(l_out_1)
    #network_output_1 = lasagne.layers.get_output(l_out_1)
    network_output_2 =  lasagne.layers.get_output(l_out_2)
    #network_output_2 = lasagne.layers.get_output(l_out_2)
    mod_y_1 = T.sqrt(T.sum(T.sqr(network_output_1), 1))
    mod_y_2 = T.sqrt(T.sum(T.sqr(network_output_2), 1))
    cosine_simi = T.sum(network_output_1*network_output_2,axis = 1)/(mod_y_1*mod_y_2)
    cost = T.mean((cosine_simi - target_values)**2)
   # cosine_sim = T.sum(network_output_1*network_output_2,axis = 1) 
    # Retrieve all parameters from the network
    all_params = lasagne.layers.get_all_params(l_out_1) + lasagne.layers.get_all_params(l_out_2)
    # Compute SGD updates for training
    print("Computing updates ...")
    updates = lasagne.updates.adagrad(cost, all_params, LEARNING_RATE)
    # Theano functions for training and computing cost
    print("Compiling functions ...")
    train = theano.function([l_in_1.input_var, l_in_2.input_var, target_values, l_mask_1.input_var,
                              l_mask_2.input_var],
                             cost,  updates=updates, on_unused_input='warn')
#     compute_cost = theano.function(
#         [l_in_1.input_var, l_in_2.input_var, target_values, l_mask_1.input_var,
#                               l_mask_2.input_var], cost, on_unused_input='warn')
#     
    test_cosine = theano.function(
        [l_in_1.input_var, l_in_2.input_var, target_values, l_mask_1.input_var,
                              l_mask_2.input_var], cosine_simi, on_unused_input='warn')
    
    train_sentence_1, train_sentence_2, cosineSimtrain, mask_train_1, mask_train_2 \
           ,test_sentence_undergoer_1,  test_sentence_undergoer_2 \
           ,cosineSimUndergoer, mask_undergoer_test_1, mask_undergoer_test_2 \
           ,test_sentence_trigger_1,  test_sentence_trigger_2 \
           ,cosineSimTrigger, mask_trigger_test_1, mask_trigger_test_2\
           ,test_sentence_enabler_1,  test_sentence_enabler_2, \
           cosineSimEnabler, mask_enabler_test_1, mask_enabler_test_2\
           ,test_sentence_result_1,  test_sentence_result_2, \
           cosineSimResult, mask_result_test_1, mask_result_test_2,\
            test_df = gen_csvdata()
           
    print("Training ...")
    try:
        for epoch in range(num_epochs):
            cost_val = train(train_sentence_1, train_sentence_2, cosineSimtrain, mask_train_1,mask_train_2 )
            #cost_val = compute_cost(train_sentence_1, train_sentence_2, cosineSimtrain, mask_train_1,mask_train_2 )
            print("Epoch {} validation cost = {}".format(epoch, cost_val))
            if epoch%100 == 0:
                cosine_undergoersim = test_cosine(test_sentence_undergoer_1,  test_sentence_undergoer_2,\
                              cosineSimUndergoer, mask_undergoer_test_1, mask_undergoer_test_2)
                cosine_enablersim = test_cosine(test_sentence_enabler_1,  test_sentence_enabler_2,\
                              cosineSimEnabler, mask_enabler_test_1, mask_enabler_test_2)
                cosine_triggersim = test_cosine(test_sentence_trigger_1,  test_sentence_trigger_2,\
                              cosineSimTrigger, mask_trigger_test_1, mask_trigger_test_2)
                cosine_resultsim = test_cosine(test_sentence_result_1,  test_sentence_result_2,\
                              cosineSimResult, mask_result_test_1, mask_result_test_2)
                test_df["newUndergoerScore"] = cosine_undergoersim
                test_df["newEnablerScore"] = cosine_enablersim
                test_df["newTriggerScore"] = cosine_triggersim
                test_df["newResultScore"] = cosine_resultsim
                test_df["avgOurScore"] = test_df.apply(averageFinalScore, axis=1)
                directory = "newresult/prediction/biRNN/"+str(epoch)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                test_df.to_csv(directory+"/cosineSimilarity.csv")
        
    except KeyboardInterrupt:
        pass    
    
if __name__ == '__main__':
       # We'll use this "validation set" to periodically check progress
    #gen_csvdata()
   # sys.argv("THEANO_FLAGS='floatX=float32,device=gpu0,nvcc.fastmath=True,exception_verbosity=high,optimizer=fast_compile'".split())
    biRNN()




