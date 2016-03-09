import cPickle
import gzip
import theano
import pdb
from fftconv import cufft, cuifft
import numpy as np
import theano.tensor as T
from theano.ifelse import ifelse
from models import *
from optimizations import *

import sys

# Warning: assumes n_batch is a divisor of number of data points
# Suggestion: preprocess outputs to have norm 1 at each time step
def main(n_iter, n_batch, n_hidden, time_steps, learning_rate, savefile, scale_penalty, use_scale, reload_progress, model, n_hidden_lstm, n_gru_lr_proj, initial_b_u):



    np.random.seed(1234)
    #import pdb; pdb.set_trace()
    # --- Set optimization params --------

    # --- Set data params ----------------
    n_input = 1
    n_output = 10
    ##### MNIST processing ################################################      

        
    # load and preprocess the data
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = cPickle.load(gzip.open("mnist.pkl.gz", 'rb'))
    n_data = train_x.shape[0]
    num_batches = n_data / n_batch

    valid_x, valid_y = test_x, test_y
    
    # shuffle data order
    inds = range(n_data)
    np.random.shuffle(inds)
    train_x = np.ascontiguousarray(train_x[inds, :time_steps])
    train_y = np.ascontiguousarray(train_y[inds])
    n_data_valid = valid_x.shape[0]
    inds_valid = range(n_data_valid)
    np.random.shuffle(inds_valid)
    valid_x = np.ascontiguousarray(valid_x[inds_valid, :time_steps])
    valid_y = np.ascontiguousarray(valid_y[inds_valid])

    # reshape x
    train_x = np.reshape(train_x.T, (time_steps, n_data, 1))
    valid_x = np.reshape(valid_x.T, (time_steps, valid_x.shape[0], 1))

    # change y to one-hot encoding
    temp = np.zeros((n_data, n_output))
    # import pdb; pdb.set_trace()
    temp[np.arange(n_data), train_y] = 1
    train_y = temp.astype('float32')

    temp = np.zeros((n_data_valid, n_output)) 
    temp[np.arange(n_data_valid), valid_y] = 1
    valid_y = temp.astype('float32')
    
    # Random permutation of pixels
    P = np.random.permutation(time_steps)
    train_x = train_x[P, :, :]
    valid_x = valid_x[P, :, :]

   #######################################################################

    # --- Compile theano graph and gradients
 
    gradient_clipping = np.float32(1)
    if (model == 'LSTM'):   
        #inputs, parameters, costs = LSTM(n_input, n_hidden_LSTM, n_output)
        inputs, parameters, costs = LSTM(n_input, n_hidden_lstm, n_output, initial_b_f = initial_b_u)
    
    #by AnvaMiba
    elif (model == 'GRU'):
        inputs, parameters, costs = GRU(n_input, n_hidden_lstm, n_output, initial_b_u = initial_b_u)
    
    #by AnvaMiba
    elif (model == 'GRU_LR'):
        inputs, parameters, costs = GRU_LR(n_input, n_hidden_lstm, n_output, n_gru_lr_proj, initial_b_u = initial_b_u)
    
    elif (model == 'complex_RNN'):
        gradient_clipping = np.float32(100000)
        inputs, parameters, costs = complex_RNN(n_input, n_hidden, n_output, scale_penalty)
    elif (model == 'complex_RNN_LSTM'):
        inputs, parameters, costs = complex_RNN_LSTM(n_input, n_hidden, n_hidden_lstm, n_output, scale_penalty)
    elif (model == 'IRNN'):
        inputs, parameters, costs = IRNN(n_input, n_hidden, n_output)
    elif (model == 'RNN'):
        inputs, parameters, costs = RNN(n_input, n_hidden, n_output)
    else:
        print >> sys.stderr, "Unsuported model:", model
        return
   
    gradients = T.grad(costs[0], parameters)

#   GRADIENT CLIPPING
    gradients = gradients[:7] + [T.clip(g, -gradient_clipping, gradient_clipping)
            for g in gradients[7:]]
 
    s_train_x = theano.shared(train_x)
    s_train_y = theano.shared(train_y)

    s_valid_x = theano.shared(valid_x)
    s_valid_y = theano.shared(valid_y)


    # --- Compile theano functions --------------------------------------------------

    index = T.iscalar('i')

    #updates, rmsprop = rms_prop(learning_rate, parameters, gradients)

    givens = {inputs[0] : s_train_x[:, n_batch * index : n_batch * (index + 1), :],
              inputs[1] : s_train_y[n_batch * index : n_batch * (index + 1), :]}

    givens_valid = {inputs[0] : s_valid_x,
                   inputs[1] : s_valid_y}
   
    
    #train = theano.function([index], [costs[0], costs[2]], givens=givens, updates=updates)
    valid = theano.function([], [costs[1], costs[2]], givens=givens_valid)

    #import pdb; pdb.set_trace()

    saved_vals = cPickle.load(file(savefile, 'rb'))
    saved_params = saved_vals['best_params']
    for i, p in enumerate(parameters):
        p.set_value(saved_params[i])
    saved_vals, saved_params = None, None

    
    [valid_cross_entropy, valid_acc] = valid()
    print >> sys.stderr, ''
    print >> sys.stderr, "TEST"
    print >> sys.stderr, "cross_entropy:", valid_cross_entropy
    print >> sys.stderr, "accurracy", valid_acc * 100
    print >> sys.stderr, '' 

if __name__=="__main__":
    kwargs = {'n_iter': 1000000,
              'n_batch': 20,
              'n_hidden': 512,
              'time_steps': 28*28,
              'learning_rate': np.float32(0.0005),
              #'savefile': '/data/lisatmp3/arjovskm/complex_RNN/2015-11-08-IRNN-permuted_mnist.pkl',
              'savefile': 'GRU_LR-permuted_mnist_128_24.pkl',
              'scale_penalty': 5,
              'use_scale': True,
              'reload_progress': True,
              #'model': 'complex_RNN',
              'model': 'GRU_LR',
              #'n_hidden_lstm': 100
              'n_hidden_lstm': 128,
              #'n_hidden_lstm': 512,
              'n_gru_lr_proj': 24,
              #'n_gru_lr_proj': 4,
              'initial_b_u': 5.0}
    main(**kwargs)
