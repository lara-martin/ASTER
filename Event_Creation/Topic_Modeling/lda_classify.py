import numpy as np
import lda
import pickle
from sklearn.feature_extraction.text import CountVectorizer

f = open('lda_model_CMU_20_1500.pkl', 'rb')
model = pickle.load(f)
f.close()

test = open('corpora/plot_summaries_noNum.txt') #CMU
tf = CountVectorizer(stop_words='english')
matrix = tf.fit_transform(test)

doc_topic_test = model.transform(matrix)
for sent_num, topics in enumerate(doc_topic_test):
	print("{} (top topic: {})".format(sent_num,topics.argmax()))
