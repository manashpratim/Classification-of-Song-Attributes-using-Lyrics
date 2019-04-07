import pickle,sys,gensim
import logging,os

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))



did = sys.argv[1]
did = str(did)

dataset 	 = pickle.load(open("../../dataset/fell_80_20_train_"+did+".pickle" , "rb"))
dataset_test = pickle.load(open("../../dataset/fell_80_20_test_"+did+".pickle" , "rb"))

trackId_2_intId = pickle.load(open("../../dataset/trackId_2_intId.pickle" , "rb"))
#artist_ratings = pickle.load(open("/scratch/kavish/fell_dataset/artist_ratings.pickle" , "rb"))

model = gensim.models.doc2vec.Doc2Vec.load('../doc2vec/model_fell_80_20_'+did+'_dbow.model')

logger.info("datasets loaded...")

op_dataset = {}

op_dataset['train'] = {}
op_dataset['test'] = {}

i = 0

for trackId in dataset:
	artist = dataset[trackId]['artist']
	lyrics = dataset[trackId]['lyrics']
	genre = dataset[trackId]['genre']
	intId = trackId_2_intId[trackId]
	vec = model.infer_vector(lyrics.split())
	'''if artist in artist_ratings:
		rating = artist_ratings[artist]
	else:
		rating = 'n/a'
	'''
	op_dataset['train'][intId] = {}
	# op_dataset['train'][intId]['artist'] = artist
	# op_dataset['train'][intId]['lyrics'] = lyrics
	op_dataset['train'][intId]['genre'] = genre
	op_dataset['train'][intId]['vector'] = vec
	# op_dataset['train'][intId]['rating'] = rating

	i += 1
	print(i,end='\r')

for trackId in dataset_test:
	artist = dataset_test[trackId]['artist']
	lyrics = dataset_test[trackId]['lyrics']
	genre = dataset_test[trackId]['genre']
	intId = trackId_2_intId[trackId]
	vec = model.infer_vector(lyrics.split())
	'''if artist in artist_ratings:
		rating = artist_ratings[artist]
	else:
		rating = 'n/a'
	'''
	op_dataset['test'][intId] = {}
	# op_dataset['test'][intId]['artist'] = artist
	# op_dataset['test'][intId]['lyrics'] = lyrics
	op_dataset['test'][intId]['genre'] = genre
	op_dataset['test'][intId]['vector'] = vec
	# op_dataset['test'][intId]['rating'] = rating

	i += 1
	print(i,end='\r')

print('\nwriting...')
pickle.dump(op_dataset, open('song_vectors_for_genre_dataset_'+did+'.pickle','wb'))
logger.info("Done")
