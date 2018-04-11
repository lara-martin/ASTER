import numpy as np
import lda
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import regex as re
import argparse

parser = argparse.ArgumentParser(description='Create a file of stories of a given topic.')
parser.add_argument('--wanted-genre', type=int, help="integer of the topic that you want to get the stories of")
parser.add_argument('--t', type=str, help="original story file")
parser.add_argument('--topic-pairs', type=str, default=None, help="file of story:topic pairs, default of None will create this file")
args = parser.parse_args()

wanted_genre = args.wanted_genre
model = pickle.load(open("lda_model_CMU_100_1500.pkl", 'rb'))
test = open(args.t, 'r')
stories = [x.strip() for x in test]
test = open(args.t, 'r')
topic_pairs = args.topic_pairs
#topic_pairs = "CMU_topic_prediction.txt" #or None


def clean_sents(text):
	original_sent = text.replace("!", ".")
	original_sent = original_sent.replace("?", ".")
	original_sent = original_sent.replace("-", " ")
	original_sent = original_sent.replace("'", " ")
	original_sent = original_sent.replace("/", " ")
	original_sent = original_sent.replace("0", " 0 ")
	original_sent = original_sent.replace("1", " 1 ")
	original_sent = original_sent.replace("2", " 2 ")
	original_sent = original_sent.replace("3", " 3 ")
	original_sent = original_sent.replace("4", " 4 ")
	original_sent = original_sent.replace("5", " 5 ")
	original_sent = original_sent.replace("6", " 6 ")
	original_sent = original_sent.replace("7", " 7 ")
	original_sent = original_sent.replace("8", " 8 ")
	original_sent = original_sent.replace("9", " 9 ")
	original_sent = re.compile(re.escape(' mr.'), re.IGNORECASE).sub(" Mr", original_sent)
	original_sent = re.compile(re.escape(' mrs.'), re.IGNORECASE).sub(" Mrs", original_sent)
	original_sent = re.compile(re.escape(' ms.'), re.IGNORECASE).sub(" Ms", original_sent)
	original_sent = re.compile(re.escape(' dr.'), re.IGNORECASE).sub(" Dr", original_sent)
	original_sent = re.compile(re.escape(' drs.'), re.IGNORECASE).sub(" Drs", original_sent)
	original_sent = re.compile(re.escape(' st.'), re.IGNORECASE).sub(" St", original_sent)
	original_sent = re.compile(re.escape(' sgt.'), re.IGNORECASE).sub(" Sgt", original_sent)
	original_sent = re.compile(re.escape(' lt.'), re.IGNORECASE).sub(" Lt", original_sent)
	original_sent = re.compile(re.escape(' fr.'), re.IGNORECASE).sub(" Fr", original_sent)
	original_sent = re.compile(re.escape(' jr.'), re.IGNORECASE).sub(" Jr", original_sent)
	original_sent = original_sent.replace("pp.", "pp")
	original_sent = original_sent.replace("pg.", "pg")
	original_sent = original_sent.replace("pgs.", "pgs")
	original_sent = original_sent.replace("pps.", "pps")
	original_sent = original_sent.replace(".com", " com")
	original_sent = original_sent.replace(".net", " net")
	original_sent = original_sent.replace(".edu", " edu")
	original_sent = original_sent.replace("www.", "www ")
	original_sent = original_sent.replace("...", " ")
	original_sent = re.sub('[@#$%^&*\(\)_\+`\-=\\/,<>:;\"\{\}\]\[]+', "", original_sent)
	original_sent = re.sub("(?<![a-zA-Z]{2,})\.","",original_sent) #removes periods in initials
	while "  " in original_sent:
		original_sent = original_sent.replace("  ", " ")
	return original_sent


if topic_pairs:
	with open("genre_"+str(wanted_genre)+"_sents.txt", "w") as outfile:
		with open(topic_pairs, 'r') as predictions:
			for pred in predictions:
				sent_num, max_topic = pred.split(":")
				if int(max_topic) == wanted_genre:
					original_sent = clean_sents(stories[int(sent_num)].strip())
					outfile.write(original_sent+"\n")
else:
	tf = CountVectorizer(stop_words='english')
	matrix = tf.fit_transform(test)
	doc_topic_test = model.transform(matrix)
	#story:topic
	with open("genre_"+str(wanted_genre)+"_sents.txt", "w") as outfile:
		for sent_num, topics in enumerate(doc_topic_test):
			max_topic = topics.argmax()
			print("{}:{}".format(sent_num,max_topic))
			if int(max_topic) == wanted_genre:
				original_sent = clean_sents(stories[int(sent_num)])
				outfile.write(original_sent+"\n")

