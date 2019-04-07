import pickle,sys
import logging,os

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))


did = sys.argv[1]
did = str(did)

sample_song_dataset = pickle.load(open('song_vectors_for_genre_dataset_'+did+'.pickle','rb'))
p = open('train_'+did+'.txt', 'w')
q = open('test_'+did+'.txt', 'w')

p.write('id')
for i in range(300):
	p.write(','+'feat'+str(i))
p.write('\n')
for songId in sample_song_dataset['train']:
	p.write(str(songId))
	for feature in sample_song_dataset['train'][songId]['vector']:
		p.write(','+str(feature))
	p.write('\n')

q.write('id')
for i in range(300):
	q.write(','+'feat'+str(i))
q.write('\n')
for songId in sample_song_dataset['test']:
	q.write(str(songId))
	for feature in sample_song_dataset['test'][songId]['vector']:
		q.write(','+str(feature))
	q.write('\n')

logger.info("Done")
