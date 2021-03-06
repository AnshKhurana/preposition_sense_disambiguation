import nltk
import numpy as np
from nltk.stem.snowball import SnowballStemmer 
stemmer = SnowballStemmer("english")
from spacy.lang.en import English
import json
import gensim
import xml.etree.ElementTree as ET 
import os
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
                prep_mask = 0 ##to make #index of word
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


def prep_feats(file_path,window_size):    
	get_glove() 
	ready_data_single = []
	ready_data   = []
	tagged_data  = []
	prep,corpus_sentences = parse_XML(file_path)
	print(len(corpus_sentences))
	oov_count = 0
	for sent in corpus_sentences:
		processed_sent = []
		sent_labels    = [sent[-1]]

        for w in sent[0]:
            word_data = dict({})
            word_data["word"]    = w.lower()
            word_data["capital"] = any(x.upper for x in w)  
                # if w_t[0].lower() not in embedding_dict.keys():
            if w.lower() not in embedding_dict.keys():
                tokens = [ str(tok) for tok in nlp(w.lower())]
                stemmed = stemmer.stem(w)
                tk_count = np.sum([1 for k in tokens if k in embedding_dict.keys() ])
                vector = np.zeros(glove_dim)
                if tk_count != 0:
                    vector = np.sum( [embedding_dict[tok] for tok in tokens if tok in embedding_dict.keys()], axis=0)/len(tokens)
                elif stemmed in  embedding_dict.keys():
                    vector = embedding_dict[stemmed]
                    if np.all( vector) ==0:
                        oov_count += 1
            else:
                vector = embedding_dict[w.lower()]
            word_data["vector"] = vector
            processed_sent.append(word_data)
        tagged_data.append(sent_labels)
        ready_data_single.append([processed_sent,sent[1]])

	for sent,indx in ready_data_single:
		new_sent = dict({})
		vl = np.stack(sent[max(0,indx-window_size):indx],axis = -1) 
        vr = np.stack(sent[indx:min(len(sent),indx+window_size)],axis = -1) 
		print(vl.shape,vr.shape())
        vi = np.concatenate((vl,vr),axis=1)
        print(vi.shape)
        new_sent["vl"]= np.mean(vl,axis=1)
        new_sent["vr"]= np.mean(vr,axis=1)
        new_sent["vi"]= 0 #todo add code to get primary pca

		ready_data.append(new_sent)
	corpus_sentences = ready_data
	corpus_labels    = tagged_data
	print("OOV count"+ str(oov_count))
	return prep,corpus_sentences, corpus_labels

def gen_train_test(window_size):
    fpath = "data_assn1/{}/Source/"
    train = dict({})
    test = dict({})
    for typr in ["Train","Test"]
        for filename in os.listdir(fpath.format(typr)):
            print(filename)
            prep,sents,labels = prep_feats(filename,window_size)
            print(prep)
            if typr=="Train":
                train[prep] = { "sents":sents , "labels":labels}
            else:
                test[prep] = { "sents":sents , "labels":labels}
    return train,test



if __name__ == '__main__':
	p,ts,tl = prep_feats('data_assn1/Train/Source/pp-about.sents.trng.xml',4)
	assert(p=="about")
    print(ts[0]['vl'].shape,tl[0])