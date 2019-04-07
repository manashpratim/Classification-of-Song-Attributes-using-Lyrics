# Classification-of-Song-Attributes-using-Lyrics
This is the official codebase of the paper[1].

# Dependencies
1) Numpy
2) Python 3
3) Tensorflow
4) Pandas
5) Scikit-Learn

# Dataset
Our dataset contains around 400,000 songs in English. We had to do extensive preprocessing to remove text that is not part of lyrics. We also had to detect and remove duplicate lyrics. Metadata about lyrics that is genre and popularity was obtained from Fell and Sporleder[2]. However, for genre and popularity prediction, we were constrained to use only a subset of dataset due to class imbalance problem. We can not publicly distribute our dataset as it might infringe copyright of the lyrics. However, our dataset is available upon request.

# Contributions
1) We are introducing SongVec (Genre/Popularity), a feature vector of songs learned using distributed representation of sentences and documents[3]. This approach overcomes the problem faced by previous works in Music Information Retrieval to have adequate domain knowledge for selecting good set of features, by learning better features automatically.
2) We have implemented deep learning based models for the song classification task. We have implemented CNN, GRU, Bi-GRU
for genre and popularity prediction task. This is the first work utilizing these approaches for song classification based on lyrics.
3) Contrary to existing work, we show that lyrics alone could be good indicators of genre and popularity.
4) We are also doing a novel analysis of content reach-ability of lyrics dataset compared to Wikipedia dataset (https://dumps.wikimedia.org/enwiki/latest/), and its potential to be a source for knowledge-based systems. We are utilizing distributed representation of words to learn word embedding[4], and using Word-Analogy test(https://www.wikidata.org/wiki/Q32127146) to get a comparison of the two datasets.

# Results
1) Genre Prediction: On an average, Genre Vector model performs the best.
2) Popularity Prediction: Deep Learning based models (especially CNN) perform better than other models.

# References
1) MP Barman, K Dahekar, A Anshuman and A Awekar "It's Only Words And Words Are All I Have"- arXiv preprint arXiv:1901.05227, 2019.
2) M. Fell and C. Sporleder. Lyrics-based analysis and classification of music. In COLING, 2014.
3) Q. Le and T. Mikolov. Distributed representations of sentences and documents. In International Conference on Machine Learning, pages 1188--1196, 2014.
4) T. Mikolov, K. Chen, G. Corrado, and J. Dean. Efficient estimation of word representations in vector space. CoRR, abs/1301.3781, 2013.
