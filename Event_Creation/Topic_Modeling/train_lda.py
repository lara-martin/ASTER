#written for python3

import numpy as np
import lda
from nltk import FreqDist
from sklearn.feature_extraction.text import CountVectorizer
import re
import operator
import pickle

f = open('../corpora/plot_summaries_noNum_clean.txt')
tf = CountVectorizer(stop_words='english')
matrix = tf.fit_transform(f)
#print(tf.vocabulary_)
vocab_tuples = sorted(tf.vocabulary_.items(), key=operator.itemgetter(1))
vocab = [x[0] for x in vocab_tuples]

model = lda.LDA(n_topics=100, n_iter=1500, random_state=1)
model.fit_transform(matrix)
topic_word = model.topic_word_
n_top_words = 25
for i, topic_dist in enumerate(topic_word):
	topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
	print('Topic {}: {}'.format(i, ' '.join(topic_words)))

output = open('lda_model_CMU_100_1500.pkl', 'wb')
pickle.dump(model, output)
output.close()

