from gensim.models import FastText
import pickle,sys,string,logging,os
from nltk.corpus import stopwords

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))


did = sys.argv[1]

test_data = pickle.load(open("../dataset/fell_80_20_test_"+did+".pickle" , "rb"))
train_data = pickle.load(open("../dataset/fell_80_20_train_"+did+".pickle" , "rb"))

def tokenize(data):
	stop_words = set(stopwords.words('english'))
	tokens = []
	for trackId in data:
		lyrics = data[trackId]['lyrics']
		lyrics = lyrics.lower()
		lyrics = lyrics.translate(str.maketrans('','',string.punctuation))
		lyrics = lyrics.split()
		filtered_lyrics = [w for w in lyrics if not w in stop_words]
		tokens.append(filtered_lyrics) 
		
	return tokens

#sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]
sentences = tokenize(train_data)
print(len(sentences))

model = FastText(sentences, min_count=5, size =300,iter =20,sample =0)
trained_vectors =model.wv
test_tokens = tokenize(test_data)
print(len(test_tokens))
dict = pickle.load(open("../dictionary/dict_{}".format(did),"rb"))
vectors = {}
count = 0
for word in dict:
	if word in trained_vectors:
		vectors[word]= trained_vectors[word]
	else:
		count += 1
i = len(dict)
print(count,len(dict),len(vectors))
for tokens in test_tokens:
	for word in tokens:
		if word in trained_vectors:
			vectors[word]= trained_vectors[word]
		else:
			count += 1
		if word not in dict:
			dict[word] = i
			i += 1

print(count,len(dict),len(vectors))
pickle.dump(vectors,open("vectors_{}".format(did),"wb"))
pickle.dump(dict,open("dict_{}".format(did),"wb"))
	
say_vector = model['say']  # get vector for word
of_vector = model['of']  # get vector for out-of-vocab word
vectors = model.wv
print(say_vector)
print(of_vector)
if 'of' in vectors:
	print (vectors['of'])
#print(model['b'])

