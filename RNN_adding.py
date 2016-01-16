import cPickle
import gzip
import theano
from fftconv import cufft, cuifft
import numpy as np
import theano.tensor as T
from theano.ifelse import ifelse
from models import *
from optimizations import *    
import argparse, timeit

import sys

# Warning: assumes n_batch is a divisor of number of data points
# Suggestion: preprocess outputs to have norm 1 at each time step
def main(n_iter, n_batch, n_hidden, time_steps, learning_rate, savefile, scale_penalty, use_scale,
         model, n_hidden_lstm, loss_function, n_gru_lr_proj, initial_b_u):

    #import pdb; pdb.set_trace()
 
    # --- Set optimization params --------
    gradient_clipping = np.float32(50000)

    # --- Set data params ----------------
    n_input = 2
    n_output = 1
  

    # --- Manage data --------------------
    n_train = 1e5
    n_test = 1e4
    num_batches = n_train / n_batch
    
    train_x = np.asarray(np.zeros((time_steps, n_train, 2)),
                         dtype=theano.config.floatX)
    

    train_x[:,:,0] = np.asarray(np.random.uniform(low=0.,
                                                  high=1.,
                                                  size=(time_steps, n_train)),
                                dtype=theano.config.floatX)
    
#    inds = np.asarray([np.random.choice(time_steps, 2, replace=False) for i in xrange(train_x.shape[1])])    
    inds = np.asarray(np.random.randint(time_steps/2, size=(train_x.shape[1],2)))
    inds[:, 1] += time_steps/2  
    
    for i in range(train_x.shape[1]):
        train_x[inds[i, 0], i, 1] = 1.0
        train_x[inds[i, 1], i, 1] = 1.0
 
    train_y = (train_x[:,:,0] * train_x[:,:,1]).sum(axis=0)
    train_y = np.reshape(train_y, (n_train, 1))

    test_x = np.asarray(np.zeros((time_steps, n_test, 2)),
                        dtype=theano.config.floatX)
    

    test_x[:,:,0] = np.asarray(np.random.uniform(low=0.,
                                                 high=1.,
                                                 size=(time_steps, n_test)),
                                dtype=theano.config.floatX)
    
    inds = np.asarray([np.random.choice(time_steps, 2, replace=False) for i in xrange(test_x.shape[1])])    
    for i in range(test_x.shape[1]):
        test_x[inds[i, 0], i, 1] = 1.0
        test_x[inds[i, 1], i, 1] = 1.0
 
   
    test_y = (test_x[:,:,0] * test_x[:,:,1]).sum(axis=0)
    test_y = np.reshape(test_y, (n_test, 1)) 


   #######################################################################

    gradient_clipping = np.float32(1)

    if (model == 'LSTM'):   
        #inputs, parameters, costs = LSTM(n_input, n_hidden_lstm, n_output, loss_function=loss_function)
        inputs, parameters, costs = LSTM(n_input, n_hidden_lstm, n_output, loss_function=loss_function, initial_b_f=initial_b_u)
        gradients = T.grad(costs[0], parameters)
        gradients = [T.clip(g, -gradient_clipping, gradient_clipping) for g in gradients]
    
    #by AnvaMiba
    elif (model == 'GRU'):   
        inputs, parameters, costs = GRU(n_input, n_hidden_lstm, n_output, loss_function=loss_function, initial_b_u=initial_b_u)
        gradients = T.grad(costs[0], parameters)
        gradients = [T.clip(g, -gradient_clipping, gradient_clipping) for g in gradients]
    #by AnvaMiba
    elif (model == 'GRU_LR'):   
        inputs, parameters, costs = GRU_LR(n_input, n_hidden_lstm, n_output, n_gru_lr_proj, loss_function=loss_function, initial_b_u=initial_b_u)
        gradients = T.grad(costs[0], parameters)
        gradients = [T.clip(g, -gradient_clipping, gradient_clipping) for g in gradients]

    elif (model == 'complex_RNN'):
        inputs, parameters, costs = complex_RNN(n_input, n_hidden, n_output, scale_penalty, loss_function=loss_function)
        if use_scale is False:
            parameters.pop()
        gradients = T.grad(costs[0], parameters)

    elif (model == 'complex_RNN_LSTM'):
        inputs, parameters, costs = complex_RNN_LSTM(n_input, n_hidden, n_hidden_lstm, n_output, scale_penalty, loss_function=loss_function)

    elif (model == 'IRNN'):
        inputs, parameters, costs = IRNN(n_input, n_hidden, n_output, loss_function=loss_function)
        gradients = T.grad(costs[0], parameters)
        gradients = [T.clip(g, -gradient_clipping, gradient_clipping) for g in gradients]

    elif (model == 'RNN'):
        inputs, parameters, costs = tanhRNN(n_input, n_hidden, n_output, loss_function=loss_function)
        gradients = T.grad(costs[0], parameters)
        gradients = [T.clip(g, -gradient_clipping, gradient_clipping) for g in gradients]

    else:
        print >> sys.stderr, "Unsuported model:", model
        return
 

   




    s_train_x = theano.shared(train_x)
    s_train_y = theano.shared(train_y)

    s_test_x = theano.shared(test_x)
    s_test_y = theano.shared(test_y)


    # --- Compile theano functions --------------------------------------------------

    index = T.iscalar('i')

    updates, rmsprop = rms_prop(learning_rate, parameters, gradients)

    givens = {inputs[0] : s_train_x[:, n_batch * index : n_batch * (index + 1), :],
              inputs[1] : s_train_y[n_batch * index : n_batch * (index + 1), :]}

    givens_test = {inputs[0] : s_test_x,
                   inputs[1] : s_test_y}
    
   
    
    train = theano.function([index], costs[0], givens=givens, updates=updates)
    test = theano.function([], costs[1], givens=givens_test)

    # --- Training Loop ---------------------------------------------------------------

    # f1 = file('/data/lisatmp3/shahamar/adding/complexRNN_400.pkl', 'rb')
    # data1 = cPickle.load(f1)
    # f1.close()
    # train_loss = data1['train_loss']
    # test_loss = data1['test_loss']
    # best_params = data1['best_params']
    # best_test_loss = data1['best_test_loss']

    # for i in xrange(len(parameters)):
    #     parameters[i].set_value(data1['parameters'][i])

    # for i in xrange(len(parameters)):
    #     rmsprop[i].set_value(data1['rmsprop'][i])
    
#    import pdb; pdb.set_trace()

    train_loss = []
    test_loss = []
    best_params = [p.get_value() for p in parameters]
    best_test_loss = 1e6
    for i in xrange(n_iter):
#        start_time = timeit.default_timer()


        if (n_iter % int(num_batches) == 0):
            #import pdb; pdb.set_trace()
            inds = np.random.permutation(int(n_train))
            data_x = s_train_x.get_value()
            s_train_x.set_value(data_x[:,inds,:])
            data_y = s_train_y.get_value()
            s_train_y.set_value(data_y[inds,:])


        mse = train(i % int(num_batches))
        train_loss.append(mse)
        print >> sys.stderr, "Iteration:", i
        print >> sys.stderr, "mse:", mse
        print >> sys.stderr, ''

        #if (i % 50==0):
        if (i % 500==0):
            mse = test()
            print >> sys.stderr, ''
            print >> sys.stderr, "TEST"
            print >> sys.stderr, "mse:", mse
            print >> sys.stderr, '' 
            test_loss.append(mse)

            if mse < best_test_loss:
                best_params = [p.get_value() for p in parameters]
                best_test_loss = mse

            
            save_vals = {'parameters': [p.get_value() for p in parameters],
                         'rmsprop': [r.get_value() for r in rmsprop],
                         'train_loss': train_loss,
                         'test_loss': test_loss,
                         'best_params': best_params,
                         'best_test_loss': best_test_loss,
                         'model': model,
                         'time_steps': time_steps}

            cPickle.dump(save_vals,
                         file(savefile, 'wb'),
                         cPickle.HIGHEST_PROTOCOL)

        
#        print 'time'
#        print timeit.default_timer() - start_time 


    
if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="training a model")
    parser.add_argument("n_iter", type=int, default=20000)
    parser.add_argument("n_batch", type=int, default=20)
    parser.add_argument("n_hidden", type=int, default=512)
    parser.add_argument("time_steps", type=int, default=200)
    parser.add_argument("learning_rate", type=float, default=0.001)
    parser.add_argument("savefile")
    parser.add_argument("scale_penalty", type=float, default=5)
    parser.add_argument("use_scale", default=True)
    parser.add_argument("model", default='complex_RNN')
    parser.add_argument("n_hidden_lstm", type=int, default=100)
    parser.add_argument("loss_function", default='MSE')
    parser.add_argument("n_gru_lr_proj", type=int, default=16)
    parser.add_argument("initial_b_u", type=float, default=1.0)
    

    args = parser.parse_args()
    dict = vars(args)

    #import pdb; pdb.set_trace()
    
    

    kwargs = {'n_iter': dict['n_iter'],
              'n_batch': dict['n_batch'],
              'n_hidden': dict['n_hidden'],
              'time_steps': dict['time_steps'],
              'learning_rate': np.float32(dict['learning_rate']),
              'savefile': dict['savefile'],
              'scale_penalty': dict['scale_penalty'],
              'use_scale': dict['use_scale'],
              'model': dict['model'],
              'n_hidden_lstm': dict['n_hidden_lstm'],
              'loss_function': dict['loss_function'],
              'n_gru_lr_proj': dict['n_gru_lr_proj'],
              'initial_b_u': dict['initial_b_u']}

    main(**kwargs)
