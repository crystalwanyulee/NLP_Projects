

import os
os.chdir('C://Users/admin/Documents/Statistical Approches of NLP/PA1_Starter_code')
import numpy as np
from collections import defaultdict
from corpus import NamesCorpus, PDTBCorpus, Name
from maxent import MaxEntClassifier
from collections import Counter
import re

def first_vocab(corpus, n_feature=50):
    
    first_vocab  = []
    
    for doc in corpus.documents['train']:
        first_vocab.extend([doc.features[0][0], doc.features[2][0]])
    
    first_vocab = list(zip(*Counter(first_vocab).most_common(n_feature)))[0]
    first_vocab = {word:i for i, word in enumerate(first_vocab)}
    
    
    for data in ['train', 'dev', 'test']:    
        n_data = len(corpus.documents[data])
        first_vector = np.zeros(n_feature*2)
                          
        for doc_id, doc in enumerate(corpus.documents[data]):
            for n, arg in enumerate([0,2]):
                first_word = doc.features[arg][0]
            
                if first_word in first_vocab.keys():
                    first_vector[first_vocab[first_word]+n*n_feature] = 1

    
            corpus.documents[data][doc_id].feature_vector = np.append(doc.feature_vector, first_vector)
    
    
    corpus.train['instance'] = np.matrix([doc.feature_vector for doc in corpus.documents['train']])
    corpus.dev['instance'] = np.matrix([doc.feature_vector for doc in corpus.documents['dev']])
    corpus.test['instance'] = np.matrix([doc.feature_vector for doc in corpus.documents['test']])
    
    corpus.vocab_dict['first'] = {key: value+corpus.num_features for key, value in first_vocab.items()}
    corpus.num_features = corpus.train['instance'].shape[1]
    
    
    return corpus


def last_vocab(corpus, n_feature=50):
    
    last_vocab = []
    
    for doc in corpus.documents['train']:
        last_vocab.extend([doc.features[0][-1], doc.features[2][-1]])

    last_vocab = list(zip(*Counter(last_vocab).most_common(n_feature)))[0]
    last_vocab = {word:i for i, word in enumerate(last_vocab)}
    
    
    for data in ['train', 'dev', 'test']:    
        n_data = len(corpus.documents[data])
        last_vector = np.zeros(n_feature*2)
                          
        for doc_id, doc in enumerate(corpus.documents[data]):
            
            for n, arg in enumerate([0,2]):
                last_word = doc.features[arg][-1]

    
                if last_word in last_vocab.keys():
                    last_vector[last_vocab[last_word]+n*n_feature] = 1
    
            corpus.documents[data][doc_id].feature_vector = np.append(doc.feature_vector, last_vector)
            
  
    
    corpus.train['instance'] = np.matrix([doc.feature_vector for doc in corpus.documents['train']])
    corpus.dev['instance'] = np.matrix([doc.feature_vector for doc in corpus.documents['dev']])
    corpus.test['instance'] = np.matrix([doc.feature_vector for doc in corpus.documents['test']])
    
    corpus.vocab_dict['last'] = {key: value+corpus.num_features for key, value in last_vocab.items()}
    corpus.num_features = corpus.train['instance'].shape[1]
    
    return corpus



def word_count(corpus, n_feature=50, count='word'):
    

    for data in ['train', 'dev', 'test']:    
        
        n_data = len(corpus.documents[data])
        word_count_vector = np.zeros(3)
                       
        for doc_id, doc in enumerate(corpus.documents[data]):
            
            args = [doc.arg1, doc.connective, doc.arg2]
            
            for n in range(3):
                if count == 'word':
                    word_count_vector[n] = len(doc.features[n])
                elif count == 'char':
                    word_count_vector[n] = len(re.findall(r'[A-Za-z0-9]', args[n]))
            

            corpus.documents[data][doc_id].feature_vector = np.append(doc.feature_vector, word_count_vector)
            
            
    corpus.train['instance'] = np.matrix([doc.feature_vector for doc in corpus.documents['train']])
    corpus.dev['instance'] = np.matrix([doc.feature_vector for doc in corpus.documents['dev']])
    corpus.test['instance'] = np.matrix([doc.feature_vector for doc in corpus.documents['test']])
    
    wc_mean = corpus.train['instance'][:, -3:].mean(axis=0)
    wc_sd = corpus.train['instance'][:, -3:].std(axis=0)
    
    def normalize(mat, v_mean, v_std):
        new_mat = (mat-v_mean)/v_std
        return new_mat
    
    corpus.train['instance'][:, -3:] = normalize(corpus.train['instance'][:, -3:], wc_mean, wc_sd)
    corpus.dev['instance'][:, -3:] = normalize(corpus.dev['instance'][:, -3:], wc_mean, wc_sd)
    corpus.test['instance'][:, -3:] = normalize(corpus.test['instance'][:, -3:], wc_mean, wc_sd)
    
    corpus.num_features = corpus.train['instance'].shape[1]
    
    return corpus



def bigram(corpus, freq = 150, n=3000):

    word_dict = corpus.vocab_dict['arg_all']   
    word_dict = Counter(word_dict).most_common(3000)
    word_dict = list(zip(*word_dict))[0]
    word_dict = {word:i for i, word in enumerate(word_dict)}
    
    bigram_mat = np.zeros((len(word_dict), len(word_dict)))
    
    
    for doc in corpus.documents['train']:
        text = doc.features[0]        
        for i in range(1, len(text)):
            if (text[i] in word_dict.keys()) & (text[i-1] in word_dict.keys()):
                bigram_mat[word_dict[text[i]], word_dict[text[i-1]]] += 1
        
    
    get_key = dict(zip(word_dict.values(),word_dict.keys()))
    
    pre, cur = np.where(bigram_mat > freq)
    
    common_bigram = defaultdict(dict)
    
    for i, pre_idx in enumerate(pre):
        pre_word = get_key[pre_idx]
        cur_word = get_key[cur[i]]
        
        if pre_word not in common_bigram.keys():
            common_bigram[pre_word] = {}
        common_bigram[pre_word][cur_word] = len(word_dict) + i
        
    corpus.vocab_dict['arg_bigram'] = common_bigram
    
    bigram_n = len(pre)
    ########################################
    
    corpus.documents['train'][0].feature_vector
    
    bigram_dict = corpus.vocab_dict['arg_bigram']
    pre_words = np.array(bigram_dict.keys())
    
    
    for arg in [0, 2]:
        for data in ['train', 'dev', 'test']:                          
            for doc_id, doc in enumerate(corpus.documents[data]):
                bigram_vector = np.zeros(bigram_n)
                
                # different arg
                arg1 = doc.features[arg]
                for i, pre_word in enumerate(np.arange(len(arg1)-1)):
                    cur_word = arg1[i+1]
                    if (pre_word in bigram_dict.keys()) & (cur_word in bigram_dict[pre_word].keys()):
                        idx = bigram_dict[pre_word][cur_word] 
                        bigram_vector[idx] = 1
                corpus.documents[data][doc_id].feature_vector = np.append(corpus.documents[data][doc_id].feature_vector, bigram_vector)
    
    
        corpus.train['instance'] = np.matrix([doc.feature_vector for doc in corpus.documents['train']])
        corpus.dev['instance'] = np.matrix([doc.feature_vector for doc in corpus.documents['dev']])
        corpus.test['instance'] = np.matrix([doc.feature_vector for doc in corpus.documents['test']])
    
        corpus.num_features = corpus.num_features + bigram_n
        
        return corpus


def trigram(corpus, freq = 70, n=10000):


    word_list = corpus.vocab_dict['arg_all']   
    len(set(word_list))
    word_list = Counter(word_list).most_common(n)
    word_list = list(zip(*word_list))[0]
    
    tri_dict = {}
    
    for doc in corpus.documents['train']:
        texts = [doc.features[0], doc.features[2]]
        for text in texts:
            for i in range(2, len(text)):
                pre, cur, nex = text[i-2], text[i-1], text[i]
                if (pre in word_list) & (cur in word_list) & (nex in word_list):
                    if pre not in tri_dict.keys():
                        tri_dict[pre] = {}
                    
                    if cur not in tri_dict[pre].keys():
                        tri_dict[pre][cur] = {}
                        
                    if nex not in tri_dict[pre][cur].keys():
                        tri_dict[pre][cur][nex] = 0
                        
                    tri_dict[pre][cur][nex] += 1
    
    
    
    common_trigram = {}
    num_trigram = 0
    
    for pre, val1 in tri_dict.items():
        for cur, val2 in val1.items():
            for nex, count in val2.items():
    
                if count > freq:
                    
                    if pre not in common_trigram.keys():
                        common_trigram[pre] = {}
                    
                    if cur not in common_trigram[pre].keys():
                        common_trigram[pre][cur] = {}
                        
                    if nex not in common_trigram[pre][cur].keys():
                        common_trigram[pre][cur][nex] = num_trigram
                        
                    num_trigram += 1
    
                   
    for data in ['train', 'dev', 'test']:
        for doc_id, doc in enumerate(corpus.documents[data]):
            texts = [doc.features[0], doc.features[2]]
            for text in texts:
                tri_vector = np.zeros(num_trigram)
                for i in range(2, len(text)):
                    pre, cur, nex = text[i-2], text[i-1], text[i]
                    if pre in common_trigram.keys():
                        if cur in common_trigram[pre].keys():
                            if nex in common_trigram[pre][cur].keys():
                                vec_idx = common_trigram[pre][cur][nex]
                                tri_vector[vec_idx] = 1
                
                corpus.documents[data][doc_id].feature_vector = np.append(doc.feature_vector, tri_vector)
                    
    
    corpus.train['instance'] = np.matrix([doc.feature_vector for doc in corpus.documents['train']])
    corpus.dev['instance'] = np.matrix([doc.feature_vector for doc in corpus.documents['dev']])
    corpus.test['instance'] = np.matrix([doc.feature_vector for doc in corpus.documents['test']])
    
    
    corpus.num_features = corpus.train['instance'].shape[1]
    
    return corpus
