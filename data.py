import nltk
import numpy as np
from nltk.stem.snowball import SnowballStemmer 
stemmer = SnowballStemmer("english")
from spacy.lang.en import English
import json
import gensim
import xml.etree.ElementTree as ET 
#from nltk.corpus import brown

nlp = English()
prefixes = sorted(json.load(open("prefix.json")), key=len, reverse=True)
suffixes = sorted(json.load(open("suffix.json")), key=len, reverse=True)
glove_dim = 50

embedding_dict = dict({})


def parse_XML(file):
    tree = ET.parse(file)
    root = tree.getroot()
    mask_sent = [] 
    
    for child in root:
        for g in child:
            if g.tag == "answer":
                y = g.attrib["senseid"]
            elif g.tag == "context":
                sent = g.text
                prep_mask = 0 ##to make
        mask_sent.append([sent,prep_mask,y])
        
    return root.attrib["item"],mask_sent


def get_breaks(corp_len,k):
	return [x*(corp_len//k) for x in range(k)] + [-1]


embedding_dict = dict({})
def get_glove():
	if len(embedding_dict.keys())==0:
		with open("glove.6B.{}d.txt".format(glove_dim), 'r') as f:
			for line in f:
				values = line.split()
				word = values[0]
				vector = np.asarray(values[1:], "float32")
				embedding_dict[word] = vector

def get_prefix_suffix(word,prefixes=prefixes,suffixes=suffixes):
		prefix   = "None"
		suffix = "None"
		word = word.lower()
		for idx,pref in enumerate(prefixes):
				if word.startswith(pref):
						prefix = pref
						break
		for idx,suff in enumerate(suffixes):
				if word.endswith(pref):
						suffix = suff
						break
		return prefix,suffix

def get_glove():
	if len(embedding_dict.keys())==0:
		with open("glove.6B.{}d.txt".format(glove_dim), 'r') as f:
			for line in f:
				values = line.split()
				word = values[0]
				vector = np.asarray(values[1:], "float32")
				embedding_dict[word] = vector



def prep_feats(file_path):
	get_glove() 
# embedding_dict = get_word2vec()
	ready_data_single = []
	ready_data   = []
	tagged_data  = []
	corpus_sentences = parse_XML(file_path)
	print(len(corpus_sentences))
	oov_count = 0
	for sent in corpus_sentences:
		processed_sent = []
		sent_labels    = []
		for w_t in sent:
			sent_labels.append(w_t[1])
			word_data = dict({})
			word_data["word"]    = w_t[0] .lower()
			word_data["capital"] = any(x.upper for x in w_t[0])  
				# if w_t[0].lower() not in embedding_dict.keys():
			if w_t[0].lower() not in embedding_dict.keys():
				tokens = [ str(tok) for tok in nlp(w_t[0].lower())]
				stemmed = stemmer.stem(w_t[0])
				tk_count = np.sum([1 for k in tokens if k in embedding_dict.keys() ])
				vector = np.zeros(glove_dim)
				if tk_count != 0:
					vector = np.sum( [embedding_dict[tok] for tok in tokens if tok in embedding_dict.keys()], axis=0)/len(tokens)
				elif stemmed in  embedding_dict.keys():
					vector = embedding_dict[stemmed]
					if np.all( vector) ==0:
						oov_count += 1
			else:
				vector = embedding_dict[w_t[0].lower()]
			pref,suff = get_prefix_suffix(w_t[0])
			# word_data["word"] = w_t[0].lower()
			word_data["pos_tag"] = w_t[2]
			word_data["prefix"] = pref 
			word_data["suffix"] = suff
			processed_sent.append(word_data)
		tagged_data.append(sent_labels)
		ready_data_single.append(processed_sent)

	for sent in ready_data_single:
		new_sent = []
		padded_sent = sent + [{"word":"PAD", "pos_tag": "PAD"},{"word":"PAD","pos_tag": "PAD"}]
		prev_vect = ["POS","POS"]
		# print(padded_sent)
		next_vect  = [padded_sent[1]['word'],padded_sent[2]['word']]
		prev_pos = ["PAD","PAD"]
		next_pos = [padded_sent[1]["pos_tag"],padded_sent[2]["pos_tag"]]
		for idx in range(len(padded_sent)-2):
			curr_word = padded_sent[idx]
			curr_word["prev_vector_0"] = prev_vect[0] 
			curr_word["prev_vector_1"] = prev_vect[1]
			curr_word["next_vector_1"] = next_vect[1] 
			curr_word["next_vector_0"] = next_vect[0]
			curr_word["prev_pos_0"] = prev_pos[0]
			curr_word["prev_pos_1"] = prev_pos[1]
			curr_word["next_pos_0"] = next_pos[0]
			curr_word["next_pos_1"] = next_pos[1]
			prev_vect = [prev_vect[1],curr_word["word"]]
			next_vect  = [next_vect[1],padded_sent[idx+2]["word"]]

			prev_pos = [prev_pos[1],curr_word["pos_tag"]]
			next_pos  = [next_pos[1],padded_sent[idx+2]["pos_tag"]]
			new_sent.append(["{}={}".format(x,curr_word[x]) for x in curr_word.keys()])
		ready_data.append(new_sent)
	corpus_sentences = ready_data
	corpus_labels    = tagged_data
	print("OOV count"+ str(oov_count))
	return corpus_sentences, corpus_labels

if __name__ == '__main__':
	ts,tl = prep_feats('../assignment2dataset/train.txt')
	print(ts[0],tl[0])