import nltk
import numpy as np
from nltk.stem.snowball import SnowballStemmer 
stemmer = SnowballStemmer("english")
from spacy.lang.en import English
import json
import xml.etree.ElementTree as ET 
import os
from sklearn.decomposition import PCA
#from nltk.corpus import brown

nlp = English()
# prefixes = sorted(json.load(open("prefix.json")), key=len, reverse=True)
# suffixes = sorted(json.load(open("suffix.json")), key=len, reverse=True)
glove_dim = 50

embedding_dict = dict({})


def parse_XML(file):
    with open(file, 'r') as f:
        text = f.read()
        text = text.replace('<context>', '<context> <left_context>')
        text = text.replace('</context>', '</right_context> </context>')
        text = text.replace('<head>', '</left_context> <head>')
        text = text.replace('</head>', '</head> <right_context>')
    
    # tree = ET.fromstring(file)
    # root = tree.getroot()
    root = ET.fromstring(text)
    mask_sent = [] 
    
    for child in root:
        # instance
        valid_sentence = True
        y = None
        for g in child:
            if g.tag == "answer":
                y = g.attrib["senseid"]
            elif g.tag == "context":
                if g[0].text is None:
                    left_context=''
                else:
                    left_context = g[0].text.strip()
                
                if g[1].text is None:
                    valid_sentence = False
                    continue
                else:
                    prep = g[1].text.strip()

                if g[2].text is None:
                    right_context=''
                else:
                    right_context = g[2].text.strip()
                sent = left_context + ' ' + prep + ' ' + right_context
                sent = sent.split()
                prep_mask = len(left_context.split())
        if valid_sentence:
            if y is None:
                y = 'unk'
            mask_sent.append([sent, prep_mask, y])    
    
    return root.attrib["item"], mask_sent


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

# def get_prefix_suffix(word,prefixes=prefixes,suffixes=suffixes):
#         prefix   = "None"
#         suffix = "None"
#         word = word.lower()
#         for idx,pref in enumerate(prefixes):
#                 if word.startswith(pref):
#                         prefix = pref
#                         break
#         for idx,suff in enumerate(suffixes):
#                 if word.endswith(pref):
#                         suffix = suff
#                         break
#         return prefix,suffix

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
    prep, corpus_sentences = parse_XML(file_path)
    # print("#Sentences in corpus", len(corpus_sentences))
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
        ready_data_single.append([processed_sent, sent[1]])

    for sent, indx in ready_data_single:
        new_sent = dict({})
        try:
            vl = np.stack([w['vector'] for w in sent[max(0 , indx-window_size):indx]], axis = -1)
        except:
            vl = np.empty((glove_dim, window_size))
        try:
            vr = np.stack([w['vector'] for w in sent[indx:min(len(sent), indx+window_size)]], axis = -1) 
        except:
            vr = np.empty((glove_dim, window_size))
        
        V_matrix = np.concatenate((vl,vr),axis=1)
        new_sent["vl"]= np.mean(vl,axis=1)
        new_sent["vr"]= np.mean(vr,axis=1)
        pca = PCA(n_components=1)
        principalComponents = pca.fit_transform(V_matrix)
        new_sent["vi"]= principalComponents[:, 0]
        new_sent['sentence'] = " ".join([w['word'] for w in sent])
        ready_data.append(new_sent)
    corpus_sentences = ready_data
    corpus_labels    = tagged_data
    # print("OOV count: ", oov_count)
    return prep, corpus_sentences, corpus_labels

def gen_train_test(window_size):
    fpath = "data_assn1/{}/Source/"
    train = dict({})
    test = dict({})
    for typr in ["Train","Test"]:
        for filename in os.listdir(fpath.format(typr)):
            # print(filename)
            prep,sents,labels = prep_feats(os.path.join(fpath.format(typr), filename), window_size)
            # print(prep)
            if typr=="Train":
                train[prep] = {"sents":sents , "labels":labels}
            else:
                test[prep] = {"sents":sents , "labels":labels}
    return train,test



if __name__ == '__main__':
    p,ts,tl = prep_feats('data_assn1/Train/Source/pp-about.sents.trng.xml', 2)
    assert(p=="about")
    gen_train_test(2)
    print(ts[0]['vl'].shape,tl[0])