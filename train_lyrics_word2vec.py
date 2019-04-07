import pickle
import sys,os,string
import gensim,logging
import multiprocessing

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))

label_option = sys.argv[1]
print("Using",label_option,"as document tags")


class MySentences(object):
    def __init__(self, datasetName):
        self.datasetName = datasetName

    def __iter__(self):
        dataset = pickle.load(open(self.datasetName , "rb"))
        for trackId in dataset:
            
            lyrics = dataset[trackId]['lyrics']

            sentences = [s for s in lyrics.split('\n') if s != '']
            for sentence in sentences:
                sentence = sentence.lower()
                sentence = sentence.translate(str.maketrans('','',string.punctuation))
                yield sentence.split()


sentences = MySentences('../fell_lyrics.pickle')

print("training model")

if sys.argv[1] == 'skipgram':
    model = gensim.models.Word2Vec(sentences,size = 300, window = 5, workers = multiprocessing.cpu_count(), min_count = 5, iter = 30, sg = 1 )
elif sys.argv[1] == 'cbow':
    model = gensim.models.Word2Vec(sentences,size = 300, window = 5, workers = multiprocessing.cpu_count(), min_count = 5, iter = 30, sg = 0 )

print("")
print("saving model")
model.save('word2vec_fell_lyrics_'+sys.argv[1]+'.model')

op = model.accuracy('questions-words.txt')
for line in op:
	print(line)
