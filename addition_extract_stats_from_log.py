# by AnvaMiba

import sys
import cPickle
import numpy as np

def main():
	if len(sys.argv) != 2:
		usage()
	
	outFs = open(sys.argv[1], 'wb')
	
	i = 0
	train_i_mse = []
	test_i_mse = []
	line = sys.stdin.readline()
	while line:
		tokens = line.split()
		if (len(tokens)) > 0 and (tokens[0] == 'Iteration:'):
			i = int(tokens[1])
			line = sys.stdin.readline()
			tokens = line.split()
			if len(tokens) != 2:
				break
			mse = float(tokens[1])
			train_i_mse.append([i, mse])
		if (len(tokens)) > 0 and (tokens[0] == 'TEST'):
			line = sys.stdin.readline()
			tokens = line.split()
			if len(tokens) != 2:
				break
			mse = float(tokens[1])
			test_i_mse.append([i, mse])
		line = sys.stdin.readline()
	
	rv_dict = {'train_i_mse': np.array(train_i_mse), 'test_i_mse': np.array(test_i_mse)}
	cPickle.dump(rv_dict, outFs, cPickle.HIGHEST_PROTOCOL)
	outFs.close()
	
def usage():
	print >> sys.stderr, 'Usage:'
	print >> sys.stderr, sys.argv[0], 'pickle_out_file'
	sys.exit(-1)

if __name__ == '__main__':
	main()

