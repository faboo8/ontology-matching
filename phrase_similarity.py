import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
warnings.filterwarnings(action='ignore', category= RuntimeWarning, module='numpy')
from gensim.models import KeyedVectors
from gensim import corpora, models, similarities


import numpy as np
import math
from nltk.corpus import stopwords, wordnet
import re
import pandas as pd



##############################################################################
##############################LOADING DATA####################################
##############################################################################

#pathToBinVectors = 'C:\\Users\DE104752\\Documents\\word2vec_pretrained\\google.bin'

#print("Loading the data file... Please wait...")
#MODEL = KeyedVectors.load_word2vec_format(pathToBinVectors, binary=True)
#print("Successfully loaded file!")


# How to call one word vector?
# MODEL['resume'] -> This will return NumPy vector of the word "resume".

##############################################################################
##############################PREPROCESSING###################################
##############################################################################


abbrevs={'excl.':'','n.e.s.':'', 'e.g.': '', 'incl.': '', 'subheading': '',
         'etc.':'', '[L.]':'', 'n.e.s':'', 'max.': 'maximum', 'kg':'', 'containing': ''}
## replace numbers???
cachedStopWords = stopwords.words("english")



def preprocessing_doc(document):
    doc_processed= []
    for phrase in document:
        phrase = phrase.lower()
        for text in abbrevs:
            phrase= phrase.replace(text,abbrevs[text])
        phrase = re.sub(r'[^\w\s]','', phrase)
        phrase = re.sub(r'[*\d]', '', phrase)
        phrase = phrase.replace('  ', ' ')
        wordsInPhrase = [word for word in phrase.split() if word not in cachedStopWords]
        doc_processed.append(wordsInPhrase)
    return doc_processed

def preprocessing_phrase(phrase):
    phrase = phrase.lower()
    for text in abbrevs:
        phrase= phrase.replace(text,abbrevs[text])
    phrase = re.sub(r'[^\w\s]','', phrase)
    phrase = re.sub(r'[*\d]', '', phrase)
    phrase = phrase.replace('  ', ' ')
    wordsInPhrase = [word for word in phrase.split() if word not in cachedStopWords]
    return wordsInPhrase, phrase



df_client = pd.read_excel('ARIBA/Warennummern Englisch.xlsx')
df1 = df_client['DESCRIP']

df1_as_list = ''
for entry in df1:
    df1_as_list += ' '+ entry
    
    
df_unspsc = pd.read_excel('ARIBA/UNSPSC_Auswahl für DBS.xlsx',skiprows=0)
df_unspsc = df_unspsc.iloc[:,0:4]
df_unspsc.columns= ['domain', 'nrel', 'code', 'DESCRIP']
df_unspsc.drop('domain', inplace=True, axis=1)
df2 = df_unspsc['DESCRIP']

df2_as_list = ''
for entry in df2:
    df2_as_list += ' ' + entry


pd_full = pd.concat([df1,df2])

document = preprocessing_doc(pd_full)
#document = [[y for x in document for y in x]]
dict1 = corpora.Dictionary(document)
corpus1 = [dict1.doc2bow(text) for text in document]
tf_model = models.TfidfModel(corpus1, id2word=dict1)
d = {dict1.get(id): value for document in tf_model[corpus1] for id, value in document}

##############################################################################
##############################################################################
##############################################################################


def sentence_similarity(sentence1, sentence2):

    sentence1,_ = preprocessing_phrase(sentence1)
    sentence2,_ = preprocessing_phrase(sentence2)
 
    # Get the synsets for the tagged words
    synsets1 = [wordnet.synsets(word)[0] for word in sentence1 if wordnet.synsets(word) != []]
    synsets2 = [wordnet.synsets(word)[0] for word in sentence2 if wordnet.synsets(word) != []] 
    
    
    # Filter out the Nones
    synsets1 = [ss for ss in synsets1 if ss]
    synsets2 = [ss for ss in synsets2 if ss]
 
    score, count = 0.0, 0.0
 
    #For each word in the first sentence
    arr_simi_score = []
    for syn1 in synsets1:
        
        for syn2 in synsets2:
            simi_score = syn1.wup_similarity(syn2)
            if simi_score is not None:
                arr_simi_score.append(simi_score)
                best = max(arr_simi_score)
                score += best
                count += 1
 
    if count != 0:# Average the values
        score /= count
    else:
        score = 0
    return score


#problem: max() excludes words that are rare but important in sentences with eg frozen!
#use weight for sentence_similarity:


#sentence_similarity is not a symmetruc function i.e. f(a.b) != f(b,a), thus define new function

def sentence_similarity_symmetric(sentence1, sentence2):
    return (sentence_similarity(sentence1, sentence2) + sentence_similarity(sentence2, sentence1))/2.0 






class PhraseVector:
    def __init__(self, phrase, MODEL, flush=True):
        self.MODEL = MODEL
        self.vector = self.PhraseToVec(phrase)
        self.phrase =phrase
        if flush == True:
            self.MODEL = None
        

    
    @staticmethod
    def LoadModel(sel ='google'):
        if sel == 'google':
            pathToBinVectors = 'C:\\Users\DE104752\\Documents\\word2vec_pretrained\\google.bin'
            print("Loading the data file... Please wait a bit... :)")
            MODEL = KeyedVectors.load_word2vec_format(pathToBinVectors, binary=True)
            print("Successfully loaded file! Yay!")
        elif sel == 'glove':
            pathToBinVectors = 'C:\\Users\DE104752\\Documents\\word2vec_pretrained\\glove_conv.txt'
            print("Loading the data file... Please wait... this is super slow")
            MODEL = KeyedVectors.load_word2vec_format(pathToBinVectors)
            print("Successfully loaded file! Hazar!")    
        return MODEL
        
    
    
    def ConvertVectorSetToVecAverageBased(self, vectorSet, ignore = []):
        # Calculates similarity between two sentences (= two  sets of vectors) based on the averages of the sets.
        #"ignore"  = "The vectors within the set that need to be ignored. If this is an empty list, nothing is ignored. 
        # returns the condensed single vector that has the same dimensionality as the other vectors within the vecotSet
        if len(ignore) == 0:
            return np.mean(vectorSet, axis = 0)
        else: 
            #bring ignore into vector format
            ignore_vectorSet= []
            for word in ignore:
                try:
                    wordVector=self.MODEL[word]
                    ignore_vectorSet.append(wordVector)
                except:
                    pass
            #stimmt das so?
            return np.dot(np.transpose(vectorSet),ignore_vectorSet)/sum(ignore)
        
    def flush_model(self):
        self.MODEL = None
        
    def PhraseToVec(self, phrase):
        cachedStopWords = stopwords.words("english")
        phrase = phrase.lower()
        for text in abbrevs:
            phrase= phrase.replace(text,abbrevs[text])
        phrase = re.sub(r'[^\w\s]','', phrase)
        phrase = re.sub(r'[*\d]', '', phrase)
        phrase = phrase.replace('  ', ' ')
        wordsInPhrase = [word for word in phrase.split() if word not in cachedStopWords]
        vectorSet = []
        for aWord in wordsInPhrase:
            try:
                wordVector=self.MODEL[aWord]
                vectorSet.append(wordVector)
            except:
                pass
        return self.ConvertVectorSetToVecAverageBased(vectorSet)
    
    def CosineSimilarity(self, other):
        #calcultes the cosine similarity based on the vectors from MODEL
        cosine_similarity = np.dot(self.vector, other.vector) / (np.linalg.norm(self.vector) * np.linalg.norm(other.vector))
        try:
            if math.isnan(cosine_similarity):
                	cosine_similarity=0
        except:
            #which error could occur?
            cosine_similarity=0		
        return cosine_similarity
    
    def WordNetSimilarity(self, other):
        return sentence_similarity_symmetric(self.phrase, other.phrase)
           
    def PhraseCompare(self,model,dictionary):
        phrase, str_phrase = preprocessing_phrase(self.phrase)
        index = similarities.MatrixSimilarity(model)
        #phrase needs to be a vector
        vec_bow = dictionary.doc2bow(str_phrase.split())
        vec_model = model[vec_bow]
        #index2 = similarities.SoftCosineSimilarity(model)
        sims= index[vec_model]
        sims = sorted(enumerate(sims))
        return sims
    
    def CombinedSimilarity(self,other, weights = [0.8,0.2]):
        if sum(weights) !=1:
            print('Weights must add to one! Normalizing...')
            weights = [weight/sum(weights) for weight in weights]
        sim1 = self.CosineSimilarity(other)
        sim2 = self.WordNetSimilarity(other)
        return (weights[0]*sim1 + weights[1]*sim2)
    
    def __str__(self):
        return self.phrase
            
            
#### document = [[word1, word2,....], [vword1, vword2,...],...]   
###do pre-processing first!!!!!
class BuildModel:
    def __init__(self, document):
        self.dict=  corpora.Dictionary(document)
        self.corpus = [self.dict.doc2bow(text) for text in document]
        self.tf_model = models.TfidfModel(self.corpus)
        
        
    def __iter__(self, corpus):
        for line in corpus:
            # corpus like: corpus = open('mycorpus.txt')
            # assume there's one document per line, tokens separated by whitespace
            # add preprocessing to line
            yield self.dict.doc2bow(line.split())

    
    def LSIModel(self,tf_model, num_topics):
        lsi = models.LsiModel(tf_model, id2word=self.dict, num_topics=num_topics)
        return lsi[tf_model]
    
    def RPModel(self,tf_model, num_topics):
        rp = models.RpModel(tf_model, num_topics=num_topics)
        return rp[tf_model]
    
    def LDAModel(self,tf_model, num_topics):
        lda = models.LdaModel(tf_model,id2word=self.dict, num_topics=num_topics)
        return lda[tf_model]
    
    def HDPModel(self,tf_model):
        hdp = models.HdpModel(tf_model,id2word=self.dict)
        return hdp[tf_model]
    
    @staticmethod    
    def AddCorpus(model, another_corpus):
        model.add_documents(another_corpus)
        
     