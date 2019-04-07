# coding: utf-8
#!/usr/bin/python

import sys, getopt, time

import operator
import pickle
import numpy as np #for euclidean distance
import pandas as pd # to read the actual dataset
from numpy import zeros, sum as np_sum, add as np_add, concatenate,     repeat as np_repeat, array, float32 as REAL, empty, ones, memmap as np_memmap,     sqrt, newaxis, ndarray, dot, vstack, dtype, divide as np_divide, integer
import scipy.sparse
import scipy.linalg
#from gensim import utils, matutils 
import logging,os

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))





blas = lambda name, ndarray: scipy.linalg.get_blas_funcs((name,), (ndarray,))[0]
blas_nrm2 = blas('nrm2', np.array([], dtype=float))
blas_scal = blas('scal', np.array([], dtype=float))

def unitvec(vec, norm='l2'):
    """
    Scale a vector to unit length. The only exception is the zero vector, which
    is returned back unchanged.
    Output will be in the same format as input
    """
    if norm not in ('l1', 'l2'):
        raise ValueError("'%s' is not a supported norm. Currently supported norms are 'l1' and 'l2'." % norm)
    if scipy.sparse.issparse(vec):
        vec = vec.tocsr()
        if norm == 'l1':
            veclen = np.sum(np.abs(vec.data))
        if norm == 'l2':
            veclen = np.sqrt(np.sum(vec.data ** 2))
        if veclen > 0.0:
            return vec / veclen
        else:
            return vec

    if isinstance(vec, np.ndarray):
        vec = np.asarray(vec, dtype=float)
        if norm == 'l1':
            veclen = np.sum(np.abs(vec))
        if norm == 'l2':
            veclen = blas_nrm2(vec)
        if veclen > 0.0:
            return blas_scal(1.0 / veclen, vec)
        else:
            return vec

    try:
        first = next(iter(vec))  # is there at least one element?
    except:
        return vec

    if isinstance(first, (tuple, list)) and len(first) == 2:  # gensim sparse format
        if norm == 'l1':
            length = float(sum(abs(val) for _, val in vec))
        if norm == 'l2':
            length = 1.0 * math.sqrt(sum(val ** 2 for _, val in vec))
        assert length > 0.0, "sparse documents must not contain any explicit zero entries"
        return ret_normalized_vec(vec, length)
    else:
        raise ValueError("unknown input type")

def similarity(d1, d2):
    """
    Compute cosine similarity between two docvecs in the trained set, specified by int index or
    string tag.
    """
    return dot(d1, d2)

def main(argv):
	input_file_name = ''
	output_file_name = ''
	try:
		opts, args = getopt.getopt(argv,"h:i:t:o:k:",["ifile=","tfile","ofile=","kval="])
	except getopt.GetoptError:
		print('top_K_sim_songs_extractor.py -i <train_file_name> -t <test_file_name> -o <output_file_name> -k <value_of_k')
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print('top_K_sim_songs_extractor.py -i <train_file_name> -t <test_file_name> -o <output_file_name> -k <value_of_k')
			sys.exit()
		elif opt in ("-i", "--ifile"):
			train_file_name = arg
			print('Input file is ', train_file_name)
		elif opt in ("-t", "--tfile"):
			test_file_name = arg
			print('Input file is ', test_file_name)
		elif opt in ("-o", "--ofile"):
			output_file_name = arg
			print('Output file is ', output_file_name)
		elif opt in ("-k", "--kval"):
			k_val = int(arg)
			print('k is ', k_val)
	
	df_train =  pd.read_csv(train_file_name)
	song_ids_train = df_train['id'].values.tolist()
	#df_train.replace('?', -99999, inplace =  True)
	df_train.drop(['id'], 1, inplace = True) 
	np_data_train = np.array(df_train.astype(float).values.tolist()) 
	#print(np_data_train)


	df_test =  pd.read_csv(test_file_name)
	song_ids_test = df_test['id'].values.tolist()
	#df_train.replace('?', -99999, inplace =  True)
	df_test.drop(['id'], 1, inplace = True) 
	np_data_test = np.array(df_test.astype(float).values.tolist()) 
	#print(np_data_train)
	for i in range(len(np_data_test)):
		np_data_test[i] = unitvec(np_data_test[i])

	for i in range(len(np_data_train)):
		np_data_train[i] = unitvec(np_data_train[i])

	song_sim_dict = {}
	for song_id in song_ids_test:
	    song_sim_dict[song_id] = []

	train_rec_count = len(song_ids_train)
	test_rec_count = len(song_ids_test)
	total_pair_count = train_rec_count*test_rec_count
	print("Need to process:", total_pair_count,"pairs")
	print("Percenatg completed: ", end='')
	sorted_song_sim_pair = []
	pair_count = 0
	t1 = time.clock()
	f = open('top_'+str(k_val)+'_similar_songs_'+output_file_name+'.txt', 'w')
	for i in range(0, test_rec_count):
		if i % 10 == 0:
			t2 = time.clock()
			print("Time taken: ", t2 - t1, "Time needed: ", ((1-(pair_count/total_pair_count))*100)*(t2 - t1),"\n")
			logger.info("Processed :",str(i),"/",str(test_rec_count)," Tests")
			#break
		sim_list = []
		for j in range(0, train_rec_count):
	    	#print(song_ids[i], "&&&", song_ids[j],"==>", similarity(np_data[i], np_data[j]))
			#song_sim_dict[song_ids_test[i]].append((song_ids_train[j], similarity(np_data_test[i], np_data_train[j])))
			sim_list.append((song_ids_train[j], similarity(np_data_test[i], np_data_train[j])))
			pair_count += 1
			#if pair_count % 1000 == 0:
				#t2 = time.clock
				#print("Time remaining: "+str((1-(pair_count/total_pair_count))*(t2-t1)/60.0))
			#	time.sleep(1)
			#	sys.stdout.write("\r%f%%" % (pair_count/total_pair_count))
			#	sys.stdout.flush()
				#break
		sorted_song_sim_pair = (song_ids_test[i], sorted(sim_list, key=operator.itemgetter(1), reverse=True))
		#print(sorted_song_sim_pair[0], sorted_song_sim_pair[1][:K])
		f.write(str(sorted_song_sim_pair[0])+'=>')
		for (song_id, sim_val) in sorted_song_sim_pair[1][:k_val]:
			f.write(str(song_id)+':'+str(sim_val)+',')
		f.write('\n')
		#pickle.dump(song_sim_dict, open('song_sim_dict.pickle','wb'))
		#pickle.dump(sorted_song_sim_pair, open('sorted_song_sim_pair.pickle','wb'))

	print('here')		
	f.close()
	# sorted_song_sim_list = []
	# i = 0
	# for key in song_sim_dict:
	#     sorted_song_sim_list.append((key, sorted(song_sim_dict[key],  key=operator.itemgetter(1), reverse=True)))
	#     #print(sorted_song_sim_list[0][0], sorted_song_sim_list[0][1][:K])
	#     f.write(str(sorted_song_sim_list[i][0])+'=>')
	#     for (song_id, sim_val) in sorted_song_sim_list[i][1][:k_val]:
	#         f.write(str(song_id)+':'+str(sim_val)+',')
	#     f.write('\n')
	#     i += 1
		# f.close()

if __name__ == "__main__":
	#print(sys.argv[1:])
	main(sys.argv[1:])
