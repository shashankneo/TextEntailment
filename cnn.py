
import numpy as np
import theano
import theano.tensor as T
import lasagne
from gensim.models import word2vec
import pandas as pd
import nltk
import numpy as np
import string
import re
import sys
from theano.compile.nanguardmode import NanGuardMode
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
LEARNING_RATE = .00000000001
# All gradients above this will be clipped
GRAD_CLIP = 100
# How often should we check the output?
EPOCH_SIZE = 100
# Number of epochs to train the net
NUM_EPOCHS = 50
n_output = 1
wordVectorDict = {}
sentenceVectors1 = []
sentenceVectors2 = []
gold_cosine_sim = []
num_epochs = 30
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
       
def getSentenceVector1_cnn(input):
    zero_word_vec = np.zeros((300,))
    leftTokens = getTokens(input["QUESTION"])
    sentVec = []
    leftTokenCount = 0
    for token in leftTokens:
        if token not in wordVectorDict:
            #print token + " in Question"
            pass
        else:
            #temp_wordVectorDict = wordVectorDict[token]
            
            #sentVec.append([[i] for i in temp_wordVectorDict]) 
            sentVec.append(wordVectorDict[token]) 
            leftTokenCount = leftTokenCount + 1
    for word in range(leftTokenCount,MAX_LENGTH):
        sentVec.append(zero_word_vec)
    #sentVec = np.reshape(sentVec, (1, sentVec.shape[0]))
    sentVec = [sentVec]
    sentenceVectors1.append(sentVec)
    
    
def getSentenceVector1(input):
    zero_word_vec = np.zeros((300,))
    leftTokens = getTokens(input["QUESTION"])
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
    sentenceVectors1.append(sentVec)
    
def getSentenceVector2_cnn(input):
    zero_word_vec = np.zeros((300,))
    leftTokens = getTokens(input["QUESTION"])
    sentVec = []
    leftTokenCount = 0
    for token in leftTokens:
        if token not in wordVectorDict:
            #print token + " in Question"
            pass
        else:
            #temp_wordVectorDict = wordVectorDict[token]
            
            #sentVec.append([[i] for i in temp_wordVectorDict]) 
            sentVec.append(wordVectorDict[token]) 

            leftTokenCount = leftTokenCount + 1
    for word in range(leftTokenCount,MAX_LENGTH):
        sentVec.append(zero_word_vec)
    #sentVec = np.reshape(sentVec, (1, sentVec.shape[0]))
    sentVec = [sentVec]
    sentenceVectors2.append(sentVec)

def getSentenceVector2(input):
    zero_word_vec = np.zeros((300,))
    leftTokens = getTokens(input["QUESTION"])
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
    sentenceVectors2.append(sentVec)

def getCosineSimOutput(input): 
    cosine_sim = input["ENT_SCORE"]
    gold_cosine_sim.append(cosine_sim)          
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
    inputs = pd.read_csv("input.csv")
    wordVec_csv = pd.read_csv("wordVectors.csv")
    wordVec_csv = wordVec_csv.drop(wordVec_csv.columns[[0]], axis=1)
    wordVec_csv.apply(formWordDictFromCsv, axis=1)
    inputs.apply(getSentenceVector1, axis=1)
    inputs.apply(getSentenceVector2, axis=1)
    inputs.apply(getCosineSimOutput, axis=1)
    
def getIndependentWordsVector_cnn():
    inputs = pd.read_csv("input.csv")
    wordVec_csv = pd.read_csv("wordVectors.csv")
    wordVec_csv = wordVec_csv.drop(wordVec_csv.columns[[0]], axis=1)
    wordVec_csv.apply(formWordDictFromCsv, axis=1)
    inputs.apply(getSentenceVector1_cnn, axis=1)
    inputs.apply(getSentenceVector2_cnn, axis=1)
    inputs.apply(getCosineSimOutput, axis=1)
    
def gen_csvdata(min_length=MIN_LENGTH, max_length=MAX_LENGTH, n_batch=N_BATCH):
    getIndependentWordsVector()
    total_length_input = len(sentenceVectors1)
    train_len = (80*total_length_input)/100
    test_len = total_length_input - train_len
    #Initialize Mask matrices
    mask_train_1 = np.zeros((train_len, max_length))
    mask_train_2 = np.zeros((train_len, max_length))
    mask_test_1 = np.zeros((test_len, max_length))
    mask_test_2 = np.zeros((test_len, max_length))
    #Get the train and test sentences, cosine similarity
    train_sentence_1 = sentenceVectors1[:train_len]
    train_sentence_2 = sentenceVectors2[:train_len]
    cosineSimtrain = gold_cosine_sim[:train_len]
    test_sentence_1 = sentenceVectors1[train_len:]
    test_sentence_2 = sentenceVectors2[train_len:]
    cosineSimtest = gold_cosine_sim[train_len:]
    
    for n in range(train_len):
        sentence_len_1 = len(train_sentence_1[n])
        mask_train_1[n,:sentence_len_1] = 1
        sentence_len_2 = len(train_sentence_2[n])
        mask_train_2[n,:sentence_len_2] = 1
       
    for n in range(test_len):
        sentence_len_1 = len(test_sentence_1[n])
        mask_test_1[n,:sentence_len_1] = 1
        sentence_len_2 = len(test_sentence_2[n])
        mask_test_2[n,:sentence_len_2] = 1       
    
    return np.array(train_sentence_2).astype(theano.config.floatX) , np.array(train_sentence_2).astype(theano.config.floatX), np.array(cosineSimtrain).astype(theano.config.floatX),\
         mask_train_1.astype(theano.config.floatX), mask_train_2.astype(theano.config.floatX) \
           ,np.array(test_sentence_1).astype(theano.config.floatX),  np.array(test_sentence_2).astype(theano.config.floatX), \
           np.array(cosineSimtest).astype(theano.config.floatX), mask_test_1.astype(theano.config.floatX), mask_test_2.astype(theano.config.floatX)


def gen_csvdata_cnn(min_length=MIN_LENGTH, max_length=MAX_LENGTH, n_batch=N_BATCH):
    getIndependentWordsVector_cnn()
    total_length_input = len(sentenceVectors1)
    train_len = (80*total_length_input)/100
    test_len = total_length_input - train_len
   
    #Get the train and test sentences, cosine similarity
    train_sentence_1 = sentenceVectors1[:train_len]
    train_sentence_2 = sentenceVectors2[:train_len]
    cosineSimtrain = gold_cosine_sim[:train_len]
    test_sentence_1 = sentenceVectors1[train_len:]
    test_sentence_2 = sentenceVectors2[train_len:]
    cosineSimtest = gold_cosine_sim[train_len:]
    
    
    
    return np.array(train_sentence_2).astype(theano.config.floatX) , np.array(train_sentence_2).astype(theano.config.floatX), np.array(cosineSimtrain).astype(theano.config.floatX),\
           np.array(test_sentence_1).astype(theano.config.floatX),  np.array(test_sentence_2).astype(theano.config.floatX), \
           np.array(cosineSimtest).astype(theano.config.floatX)


def gen_data(min_length=MIN_LENGTH, max_length=MAX_LENGTH, n_batch=N_BATCH):
    # Generate X - we'll fill the last dimension later
    X = np.concatenate([np.random.uniform(size=(n_batch, max_length, 1)),
                        np.zeros((n_batch, max_length, 1))],
                       axis=-1)
    mask = np.zeros((n_batch, max_length))
    y = np.zeros((n_batch,))
    # Compute masks and correct values
    for n in range(n_batch):
        # Randomly choose the sequence length
        length = np.random.randint(min_length, max_length)
        # Make the mask for this sample 1 within the range of length
        mask[n, :length] = 1
        # Zero out X after the end of the sequence
        X[n, length:, 0] = 0
        # Set the second dimension to 1 at the indices to add
        X[n, np.random.randint(length/10), 1] = 1
        X[n, np.random.randint(length/2, length), 1] = 1
        # Multiply and sum the dimensions of X to get the target value
        y[n] = np.sum(X[n, :, 0]*X[n, :, 1])
    # Center the inputs and outputs
    X -= X.reshape(-1, 2).mean(axis=0)
    y -= y.mean()
    return (X.astype(theano.config.floatX), y.astype(theano.config.floatX),
            mask.astype(theano.config.floatX))
def RNN():
    # First, we build the network, for first sentence starting with an input layer
    # Recurrent layers expect input of shape
    # (batch size, max sequence length, number of features)
    #Giving the batch size as None because we are still experimenting with the 
    # meaning and true usage of the parameter
    #Sequence length corresponds to time steps but this would be variable and it would depend
    #upon the input length so lets give it as None
    #Number of features are 300 because each word is a vector of 300 dimensions
    
    W = lasagne.init.HeUniform()
    l_in_1 = lasagne.layers.InputLayer(shape=(None, MAX_LENGTH, 300))
    l_mask_1 = lasagne.layers.InputLayer(shape=(None, MAX_LENGTH))
    l_forward_1 = lasagne.layers.RecurrentLayer(
        l_in_1, N_HIDDEN, mask_input=l_mask_1, grad_clipping=GRAD_CLIP,
        W_in_to_hid=W,
        W_hid_to_hid=lasagne.init.HeUniform(),
        nonlinearity=lasagne.nonlinearities.tanh)
    l_out_1 = lasagne.layers.SliceLayer(l_forward_1, -1, 1)
    #l_out_1 = lasagne.layers.DenseLayer(l_forward_1, num_units=n_output)
    
    l_in_2 = lasagne.layers.InputLayer(shape=(None, MAX_LENGTH, 300))
    l_mask_2 = lasagne.layers.InputLayer(shape=(None, MAX_LENGTH))
    l_forward_2 = lasagne.layers.RecurrentLayer(
        l_in_2, N_HIDDEN, mask_input=l_mask_2, grad_clipping=GRAD_CLIP,
        W_in_to_hid=lasagne.init.HeUniform(),
        W_hid_to_hid=lasagne.init.HeUniform(),
        nonlinearity=lasagne.nonlinearities)
    l_out_2 = lasagne.layers.SliceLayer(l_forward_2, -1, 1)
    #l_out_2 = lasagne.layers.DenseLayer(l_forward_2, num_units=n_output)
    
    #target cosine similarity of the pair of sentence
    target_values = T.vector('target_output')
    network_output_1 = lasagne.layers.get_output(l_out_1)
    #network_output_1 = lasagne.layers.get_output(l_out_1)
    network_output_2 =  lasagne.layers.get_output(l_out_2)
    #network_output_2 = lasagne.layers.get_output(l_out_2)
    cost = T.mean((T.sum(network_output_1*network_output_2,axis = 1) - target_values)**2)
    cosine_sim = network_output_1 * network_output_2
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
    compute_cost = theano.function(
        [l_in_1.input_var, l_in_2.input_var, target_values, l_mask_1.input_var,
                              l_mask_2.input_var], cost, on_unused_input='warn')
    
    test_cosine = theano.function(
        [l_in_1.input_var, l_in_2.input_var, target_values, l_mask_1.input_var,
                              l_mask_2.input_var], cosine_sim, on_unused_input='warn')
    
    train_sentence_1, train_sentence_2, cosineSimtrain, mask_train_1, mask_train_2 \
           ,test_sentence_1,  test_sentence_2, cosineSimtest, mask_test_1, mask_test_2 = gen_csvdata()
           
    print("Training ...")
    try:
        for epoch in range(NUM_EPOCHS):
            train(train_sentence_1, train_sentence_2, cosineSimtrain, mask_train_1,mask_train_2 )
            cost_val = compute_cost(train_sentence_1, train_sentence_2, cosineSimtrain, mask_train_1,mask_train_2 )
            print("Epoch {} validation cost = {}".format(epoch, cost_val))
        test_cosine(test_sentence_1, test_sentence_2, test_cosine, mask_test_1, mask_test_2)
    except KeyboardInterrupt:
        pass
    
def CNN():
    l_in_1 = lasagne.layers.InputLayer(shape=(None, 1, MAX_LENGTH, 300))
    l_conv_1 = lasagne.layers.Conv2DLayer(
            l_in_1, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    
    l_pool_1 = lasagne.layers.MaxPool2DLayer(l_conv_1, pool_size=(2, 2))
    
    l_conv_second_1 = lasagne.layers.Conv2DLayer(
            l_pool_1, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
            
    l_pool_second_1 = lasagne.layers.MaxPool2DLayer(l_conv_second_1, pool_size=(2, 2))

    l_dense_1 = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(l_pool_second_1, p=.5),
            num_units=1024,
            nonlinearity=lasagne.nonlinearities.rectify)
            
            
    l_in_2 = lasagne.layers.InputLayer(shape=(None, 1, MAX_LENGTH, 300))
    l_conv_2 = lasagne.layers.Conv2DLayer(
            l_in_2, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    
    l_pool_2 = lasagne.layers.MaxPool2DLayer(l_conv_2, pool_size=(2, 2))
    
    l_conv_second_2 = lasagne.layers.Conv2DLayer(
            l_pool_2, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
            
    l_pool_second_2 = lasagne.layers.MaxPool2DLayer(l_conv_second_2, pool_size=(2, 2))

    l_dense_2 = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(l_pool_second_2, p=.5),
            num_units=1024,
            nonlinearity=lasagne.nonlinearities.rectify)
  
    l_out_1 = l_dense_1
    l_out_2 = l_dense_2
          
    target_values = T.vector('target_output')
    network_output_1 = lasagne.layers.get_output(l_out_1)
    #network_output_1 = lasagne.layers.get_output(l_out_1)
    network_output_2 =  lasagne.layers.get_output(l_out_2)
    #network_output_2 = lasagne.layers.get_output(l_out_2)
    cost = lasagne.objectives.categorical_crossentropy(T.sum(network_output_1*network_output_2,axis = 1), target_values)
    cost = cost.mean()
    cosine_sim = network_output_1 * network_output_2
    # Retrieve all parameters from the network
    all_params = lasagne.layers.get_all_params(l_out_1) + lasagne.layers.get_all_params(l_out_2)
    # Compute SGD updates for training
    print("Computing updates ...")
    updates = lasagne.updates.nesterov_momentum(
            cost, all_params, learning_rate=0.000000001, momentum=0.5)
    # Theano functions for training and computing cost
    print("Compiling functions ...")
    train = theano.function([l_in_1.input_var, l_in_2.input_var, target_values],
                            cost,  updates=updates, on_unused_input='warn', mode=NanGuardMode(nan_is_error=True, 
                            inf_is_error=True, big_is_error=True))
    compute_cost = theano.function(
        [l_in_1.input_var, l_in_2.input_var, target_values], cost, on_unused_input='warn',
        mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))
    
    test_cosine = theano.function(
        [l_in_1.input_var, l_in_2.input_var, target_values], cosine_sim, on_unused_input='warn')
    
    train_sentence_1, train_sentence_2, cosineSimtrain,\
           test_sentence_1,  test_sentence_2, cosineSimtest = gen_csvdata_cnn()
           
    print("Training ...")
    try:
        for epoch in range(NUM_EPOCHS):
            train(train_sentence_1, train_sentence_2, cosineSimtrain )
            cost_val = compute_cost(train_sentence_1, train_sentence_2, cosineSimtrain )
            print("Epoch {} validation cost = {}".format(epoch, cost_val))
        test_cosine(test_sentence_1, test_sentence_2, test_cosine)
    except KeyboardInterrupt:
        pass
    cosine_sim = test_cosine(test_sentence_1,  test_sentence_2, cosineSimtest)
    #x = pd.Series(cosine_sim[0])
    #test_df["newCosineSimilarity"] = cosine_sim[0]
    #test_df.to_csv("cosineSimilariry.csv")
    
    
    
    
if __name__ == '__main__':
       # We'll use this "validation set" to periodically check progress
   # gen_csvdata()
   # sys.argv("THEANO_FLAGS='floatX=float32,device=gpu0,nvcc.fastmath=True,exception_verbosity=high,optimizer=fast_compile'".split())
    CNN()



