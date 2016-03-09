# by AnvaMiba

import sys
import cPickle
import numpy as np
import theano
from sklearn.utils import check_random_state

n_train = int(1e5)
n_test  = int(1e4)

def main():
	if len(sys.argv) != 3:
		usage()
	
	time_steps = int(sys.argv[1])
        outFile = sys.argv[2]
        outFs = open(outFile, 'w')

	n = n_train + n_test
	rng = check_random_state(0)

	x = np.zeros((time_steps+20, n, 10), dtype=theano.config.floatX)
	y = np.zeros((time_steps+20, n, 10), dtype=theano.config.floatX)
	
	for i in xrange(n):
	        seq_len = rng.random_integers(1, 10)
	        ids = rng.random_integers(0, 7, size=(seq_len,))
	        x[seq_len:seq_len+time_steps-1, i, 8] = 1.0
	        x[seq_len+time_steps-1, i, 9] = 1.0
	        x[seq_len+time_steps:, i, 8] = 1.0
                y[:seq_len+time_steps, i, 8] = 1.0
		for t in xrange(seq_len):
			x[t, i, ids[t]] = 1.0
                        y[seq_len+time_steps+t, i, ids[t]] = 1.0
                y[2*seq_len+time_steps:, i, 8] = 1.0
                

        train_x = x[:, :n_train, :]
        train_y = y[:, :n_train, :]
        test_x  = x[:, n_train:, :]
        test_y  = y[:, n_train:, :]
        data_dict = {'train_x':train_x, 'train_y':train_y, 'test_x':test_x, 'test_y':test_y}
        cPickle.dump(data_dict, outFs, cPickle.HIGHEST_PROTOCOL)
        outFs.close()

def usage():
	print >> sys.stderr, 'Usage:'
	print >> sys.stderr, sys.argv[0], 'time_steps out_file'
	sys.exit(-1)

if __name__ == '__main__':
	main()

