import pickle
import sys,os,string,codecs,pickle
import gensim,logging
import multiprocessing

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))

did = sys.argv[2]
did = str(did)

logger.info("Dataset Id : "+did)
logger.info("Loaded : fell_80_20_train_"+did+".pickle")

class MySentences(object):
    def __init__(self, datasetName):
        self.datasetName = datasetName

    def __iter__(self):

        dataset = pickle.load(open("../../dataset/fell_80_20_train_"+did+".pickle" , "rb"))
        #i=1
        for trackId in dataset:
            #if(1==1):
            lyrics      = dataset[trackId]['lyrics']
            lyrics = lyrics.lower()
            lyrics = lyrics.translate(str.maketrans('','',string.punctuation))
            #print(lyrics)
            genre_label = dataset[trackId]['genre']
            genre_label = 'GENRE_'+genre_label
            song = gensim.models.doc2vec.LabeledSentence(words=lyrics.split(), tags=[genre_label])
            # log.info("Artist :: "+str(artist_name))
            yield song
            #i += 1
              '''
              sentences = [s for s in lyrics.split('\n') if s != '']
              for sentence in sentences:
                sentence = sentence.lower()
                sentence = sentence.translate(str.maketrans('','',string.punctuation))
                # yield sentence.split()
                song = gensim.models.doc2vec.LabeledSentence(words=sentence.split(), tags=[genre_label])
                # log.info("Artist :: "+str(artist_name))
                yield song
              '''	   

sentences = MySentences('')
#for ab in sentences:
#	print(ab)
#print (len(sentences))
print("training model")

if sys.argv[1] == 'dm':
    model = gensim.models.Doc2Vec (sentences,size = 300, window = 10, workers = multiprocessing.cpu_count(),min_count= 5, iter = 20, dm=1)
elif sys.argv[1] == 'dbow':
    model = gensim.models.Doc2Vec (sentences,size = 300, window = 10, workers = multiprocessing.cpu_count(),min_count = 5, iter = 20, dm=0)

print("")
print("saving model")
model.save('model_fell_80_20_'+did+'_dbow.model')

