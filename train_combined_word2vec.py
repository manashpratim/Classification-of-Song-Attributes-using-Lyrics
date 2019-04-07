
import pickle
import sys,os,codecs,string
import gensim,logging
import multiprocessing

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))

input_Wikipedia_file = sys.argv[1]
output_file = sys.argv[2]


class MySentences(object):
    def __init__(self, datasetName):
        self.datasetName = datasetName

    def __iter__(self):
        with codecs.open(self.datasetName, "r" , encoding='utf-8' , errors='ignore') as inFile:
            for line in inFile:
                if line == '':
                    continue
                # While extracting the tokens in single file we append a summary at the end, below line avoids that while training.
                if line[0:11] == '####Summary':
                    continue
                sentences = line.split('.')
                for sentence in sentences:
                    #print("Hello")
                    sentence = sentence.lower()
                    sentence = sentence.translate(str.maketrans('','',string.punctuation))
                    yield sentence.split()
        #print("Here")
        dataset = pickle.load(open("../fell_lyrics.pickle" , "rb"))
        for trackId in dataset:
            
            lyrics = dataset[trackId]['lyrics']

            sentences = [s for s in lyrics.split('\n') if s != '']
            for sentence in sentences:
                sentence = sentence.lower()
                sentence = sentence.translate(str.maketrans('','',string.punctuation))
                yield sentence.split() 


sentences = MySentences(input_Wikipedia_file)

print("training model")

# if sys.argv[1] == 'skipgram':
#     model = gensim.models.Word2Vec(sentences,size = 300, window = 10, workers = multiprocessing.cpu_count(), min_count = 5, iter = 30, sg = 1 )
# elif sys.argv[1] == 'cbow':
#     model = gensim.models.Word2Vec(sentences,size = 300, window = 5, workers = multiprocessing.cpu_count(), min_count = 5, iter = 30, sg = 0 )

model = gensim.models.Word2Vec(sentences,size = 300, window = 10, workers = multiprocessing.cpu_count(), min_count = 5, iter = 30, sg = 0 )

print("")
print("saving model")
model.save(output_file+'_cbow.model')

op = model.accuracy('questions-words.txt')
for line in op:
	print(line)
