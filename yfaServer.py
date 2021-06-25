# Server script for youtube focus assist google chrome extension
import flask
from flask_cors import CORS
import re
import spacy
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

server = flask.Flask(__name__)

nlp = spacy.load("en_core_web_sm")

def simplify(txt):
	doc = nlp(txt)
	proc_txt = []
	for word in doc:
		if word.is_stop == False and word.is_alpha == True and word.pos_ in ["NOUN", "PROPN", "ADJ"]:
			proc_txt.append(word.lemma_.lower())
	
	proc_txt = " ".join(proc_txt)
	return proc_txt

def preprocess(txt):
	txt = re.sub('\xFF|\uFFFF|\t|\n|\v|\f|\r|\0|\+|\000|https?:\/\/\S+|[^\u0041-\u005a\u0061-\u007a\u0030-\u0039.!&\'",.? ]', " ", txt)
	txt = re.sub(r' +', ' ', txt)
	txt = simplify(txt)

	return txt

def similarity(txt1, txt2):
	vectorizer = TfidfVectorizer()
	X = vectorizer.fit_transform([txt1, txt2])
	X_sparse = X.toarray()
	root_x = np.squeeze(X_sparse[0])
	x = np.squeeze(X_sparse[1])

	similarity_tfidf = np.dot(root_x, x)/(np.linalg.norm(root_x)*np.linalg.norm(x))
	return similarity_tfidf

def getScores(mainInfo, infos):
	# mainInfo => dict with id and vidText or mainVid
	# infos => list of dicts with id and vidText or mainVid

	scores = []
	mainTxt = preprocess(mainInfo["vidText"])

	for vid in infos:
		vidTxt = preprocess(vid["vidText"])
		scores.append(similarity(mainTxt, vidTxt))
	
	return scores

CORS(server)

@server.route("/get-scores", methods=["POST"])
def runScript():
	data = flask.request.get_json()
	
	try:
		return flask.jsonify(scores = getScores(data["mainInfo"], data["infos"]))
	except Exception as e:
		return e





