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
import argparse, timeit

import sys
from theano.printing import Print

def clip_gradients_norm(gradients, threshold, parameters, fix_nan = False):
	gradient_sqr_vec = T.concatenate([T.sqr(g.flatten()) for g in gradients])
	gradient_norm = T.sqrt(gradient_sqr_vec.sum())
	rescale = T.maximum(gradient_norm, threshold)
	if fix_nan:
		isnan = T.or_(T.isnan(gradient_norm), T.isinf(gradient_norm))
	else:
		isnan = None
	rv = []
	for i, g in enumerate(gradients):
		if fix_nan:
			alt_g = 0.1 * parameters[i]
			print_alt_g = Print("NaN detected! Fixing with pseudogradient with mean:", ["mean"])(alt_g)
			new_g = T.switch(isnan, print_alt_g, g / rescale)
		else:
			new_g = g / rescale
		rv.append(new_g)
	return rv

# Warning: assumes n_batch is a divisor of number of data points
# Suggestion: preprocess outputs to have norm 1 at each time step
def main(datafile, n_iter, n_batch, n_hidden, time_steps, learning_rate, savefile, scale_penalty, use_scale,
         model, n_hidden_lstm, loss_function, cost_every_t, n_gru_lr_proj, initial_b_u, load_iter):


    # --- Manage data --------------------
    #f = file('/u/shahamar/complex_RNN/trainingRNNs-master/memory_data_300.pkl', 'rb')
    #f = file('./memory_data_20.pkl', 'rb')
    #f = file('./memory_data_500.pkl', 'rb')
    f = file(datafile, 'rb')
    dict = cPickle.load(f)
    f.close()


    train_x = dict['train_x'] 
    train_y = dict['train_y']
    test_x = dict['test_x'] 
    test_y = dict['test_y'] 


    #import pdb; pdb.set_trace()
    #cPickle.dump(dict, file('/u/shahamar/complex_RNN/trainingRNNs-master/memory_data.pkl', 'wb'))

    #import pdb; pdb.set_trace()

    n_train = train_x.shape[1]
    n_test = test_x.shape[1]
    n_input = train_x.shape[2]
    n_output = train_y.shape[2]

    num_batches = int(n_train / n_batch)
    time_steps = train_x.shape[0]
    
   #######################################################################

    gradient_clipping = np.float32(1)

    if (model == 'LSTM'):   
        inputs, parameters, costs = LSTM(n_input, n_hidden_lstm, n_output,
                                         out_every_t=cost_every_t, loss_function=loss_function)
        gradients = T.grad(costs[0], parameters)
        gradients = [T.clip(g, -gradient_clipping, gradient_clipping) for g in gradients]
    
    elif (model == 'GRU'): 
        inputs, parameters, costs = GRU(n_input, n_hidden_lstm, n_output, 
                                         out_every_t=cost_every_t, loss_function=loss_function, initial_b_u=initial_b_u)
        gradients = T.grad(costs[0], parameters)
        gradients = [T.clip(g, -gradient_clipping, gradient_clipping) for g in gradients]
   
    elif (model == 'GRU_LR'):
        inputs, parameters, costs = GRU_LR(n_input, n_hidden_lstm, n_output, n_gru_lr_proj,
                                         out_every_t=cost_every_t, loss_function=loss_function, initial_b_u=initial_b_u)
        gradients = T.grad(costs[0], parameters)
        #gradients = [T.clip(g, -gradient_clipping, gradient_clipping) for g in gradients]
        gradients = clip_gradients_norm(gradients, gradient_clipping, parameters, fix_nan = True)
    
    elif (model == 'GRU_LRDiag'):
        inputs, parameters, costs = GRU_LRDiag(n_input, n_hidden_lstm, n_output, n_gru_lr_proj,
                                         out_every_t=cost_every_t, loss_function=loss_function, initial_b_u=initial_b_u)
        gradients = T.grad(costs[0], parameters)
        #gradients = [T.clip(g, -gradient_clipping, gradient_clipping) for g in gradients]
        gradients = clip_gradients_norm(gradients, gradient_clipping, parameters, fix_nan = True)

    elif (model == 'complex_RNN'):
        inputs, parameters, costs = complex_RNN(n_input, n_hidden, n_output, scale_penalty,
                                                out_every_t=cost_every_t, loss_function=loss_function)
        if use_scale is False:
            parameters.pop()
        gradients = T.grad(costs[0], parameters)

    elif (model == 'complex_RNN_LSTM'):
        inputs, parameters, costs = complex_RNN_LSTM(n_input, n_hidden, n_hidden_lstm, n_output, scale_penalty,
                                                     out_every_t=cost_every_t, loss_function=loss_function)

    elif (model == 'IRNN'):
        inputs, parameters, costs = IRNN(n_input, n_hidden, n_output,
                                         out_every_t=cost_every_t, loss_function=loss_function)
        gradients = T.grad(costs[0], parameters)
        gradients = [T.clip(g, -gradient_clipping, gradient_clipping) for g in gradients]

    elif (model == 'RNN'):
        inputs, parameters, costs = tanhRNN(n_input, n_hidden, n_output,
                                            out_every_t=cost_every_t, loss_function=loss_function)
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

    #updates, rmsprop = rms_prop(learning_rate, parameters, gradients)
    updates, rmsprop = rms_prop(learning_rate, parameters, gradients, epsilon=1e-8)

    givens = {inputs[0] : s_train_x[:, n_batch * index : n_batch * (index + 1), :],
              inputs[1] : s_train_y[:, n_batch * index : n_batch * (index + 1), :]}

    givens_test = {inputs[0] : s_test_x,
                   inputs[1] : s_test_y}
    
   
    
    #train = theano.function([index], costs[0], givens=givens, updates=updates)
    train = theano.function([index], [costs[0], costs[2]], givens=givens, updates=updates)
    #test = theano.function([], costs[1], givens=givens_test)
    test = theano.function([], [costs[1], costs[2]], givens=givens_test)

    # --- Training Loop ---------------------------------------------------------------

    # f1 = file('/data/lisatmp3/shahamar/memory/RNN_100.pkl', 'rb')
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

    corrected_accuracy = lambda a: 4.0 / 5.0 * (time_steps * (a - 1.0) + 20.0 * a - 10.0)

    train_loss = []
    test_loss = []
    best_params = [p.get_value() for p in parameters]
    best_test_loss = 1e6

    if load_iter > 0:
	loaded_data = cPickle.load(file(savefile, 'rb'))
        loaded_parameters = loaded_data['parameters']
	for j, p in enumerate(parameters):
        	p.set_value(loaded_parameters[j])
        loaded_rmsprop = loaded_data['rmsprop']
        for j, r in enumerate(rmsprop):
		r.set_value(loaded_rmsprop[j])
        train_loss = loaded_data['train_loss']
	test_loss = loaded_data['test_loss']
        best_params = loaded_data['best_params']
        best_test_loss =  loaded_data['best_test_loss']
        model = loaded_data['model']
        time_steps = loaded_data['time_steps']

    for i in xrange(load_iter, n_iter):
#        start_time = timeit.default_timer()
     #   pdb.set_trace()

        if (n_iter % int(num_batches) == 0):
            #import pdb; pdb.set_trace()
            inds = np.random.permutation(int(n_train))
            data_x = s_train_x.get_value()
            s_train_x.set_value(data_x[:,inds,:])
            data_y = s_train_y.get_value()
            s_train_y.set_value(data_y[:,inds,:])


        mse, accuracy = train(i % int(num_batches))
        train_loss.append(mse)
        print >> sys.stderr, "Iteration:", i
        #print >> sys.stderr, "mse:", mse
        print >> sys.stderr, "ce:", mse, "accuracy:", accuracy, "corrected accuracy:", corrected_accuracy(accuracy), "/ 8"
        print >> sys.stderr, ''

        #if (i % 50==0):
        if (i % 500==0):
            mse, accuracy = test()
            print >> sys.stderr, ''
            print >> sys.stderr, "TEST"
            #print >> sys.stderr, "mse:", mse
            print >> sys.stderr, "ce:", mse, "accuracy:", accuracy, "corrected accuracy:", corrected_accuracy(accuracy), " / 8"
            print >> sys.stderr, ''
            test_loss.append(mse)

            if mse < best_test_loss:
                best_params = [p.get_value() for p in parameters]
                best_test_loss = mse
                print >> sys.stderr, "BEST!"

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
    parser.add_argument("datafile")
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
    parser.add_argument("cost_every_t", default=False)
    parser.add_argument("n_gru_lr_proj", type=int, default=16)
    parser.add_argument("initial_b_u", type=float, default=1.0)
    parser.add_argument("load_iter", type=int, default=0)


    args = parser.parse_args()
    dict = vars(args)

    #import pdb; pdb.set_trace()
    
    

    kwargs = {'datafile': dict['datafile'],
              'n_iter': dict['n_iter'],
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
              'cost_every_t': dict['cost_every_t'],
              'n_gru_lr_proj': dict['n_gru_lr_proj'],
              'initial_b_u': dict['initial_b_u'],
              'load_iter': dict['load_iter']}

    main(**kwargs)
