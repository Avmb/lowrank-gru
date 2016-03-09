# by AnvaMiba

import sys
import cPickle
import numpy as np

def main():
	if len(sys.argv) != 2:
		usage()
	
	outFs = open(sys.argv[1], 'wb')
	
	i = 0
	train_i_ce_acc_cacc = []
	test_i_ce_acc_cacc = []
	line = sys.stdin.readline()
	while line:
		tokens = line.split()
		if (len(tokens)) > 0 and (tokens[0] == 'Iteration:'):
			i = int(tokens[1])
			line = sys.stdin.readline()
			tokens = line.split()
			if len(tokens) != 9:
				break
			ce = float(tokens[1])
			acc = float(tokens[3])
			cacc = float(tokens[6])
			train_i_ce_acc_cacc.append([i, ce, acc, cacc])
		if (len(tokens)) > 0 and (tokens[0] == 'TEST'):
			line = sys.stdin.readline()
			tokens = line.split()
			if len(tokens) != 9:
				break
			ce = float(tokens[1])
			acc = float(tokens[3])
			cacc = float(tokens[6])
			test_i_ce_acc_cacc.append([i, ce, acc, cacc])
		line = sys.stdin.readline()
	
	rv_dict = {'train_i_ce_acc_cacc': np.array(train_i_ce_acc_cacc), 'test_i_ce_acc_cacc': np.array(test_i_ce_acc_cacc)}
	cPickle.dump(rv_dict, outFs, cPickle.HIGHEST_PROTOCOL)
	outFs.close()
	
def usage():
	print >> sys.stderr, 'Usage:'
	print >> sys.stderr, sys.argv[0], 'pickle_out_file'
	sys.exit(-1)

if __name__ == '__main__':
	main()

