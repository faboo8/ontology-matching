# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 13:59:49 2018

@author: DE104752
"""

import phrase_similarity as phsim
import pandas as pd
from gensim import corpora, models
import tqdm
import pickle
from multiprocessing import Pool
import os






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

document = phsim.preprocessing_doc(pd_full)
#document = [[y for x in document for y in x]]
dict1 = corpora.Dictionary(document)
corpus1 = [dict1.doc2bow(text) for text in document]
tf_model = models.TfidfModel(corpus1, id2word=dict1)
d = {dict1.get(id): value for document in tf_model[corpus1] for id, value in document}
global d2
d2 = {}
def process_ont(el):
    vec1 = phsim.PhraseVector(el)
    list1, list2, list3= [], [], []
    print('Finding matches for {}'.format(vec1.phrase))
    for line in df2:
        line_obj= phsim.PhraseVector(line)
        list1.append(vec1.CombinedSimilarity(line_obj))
        list2.append(vec1.CosineSimilarity(line_obj))
        list3.append(vec1.WordNetSimilarity(line_obj))
    
    list1 = pd.DataFrame(list1, columns= ['sim_score']).sort_values(by ='sim_score', ascending=False).head(5)
    list2 = pd.DataFrame(list2, columns= ['sim_score']).sort_values(by ='sim_score', ascending=False).head(5)
    list3 = pd.DataFrame(list3, columns= ['sim_score']).sort_values(by ='sim_score', ascending=False).head(5)
    
    list1['name'] = [df2[x]  for x in list1.index]
    list2['name'] = [df2[x]  for x in list2.index]
    list3['name'] = [df2[x]  for x in list3.index]
    d2[el] = [list1, list2, list3]
if __name__ == '__main__':
    sel = input('Do you want to use word vectors from google or glove? \n')
    MODEL = phsim.PhraseVector.LoadModel(sel)
    pool = Pool(os.cpu_count() - 1)                         # Create a multiprocessing Pool
    pool.map(process_ont, df1[:3]) 
    with open('filename.pickle', 'wb') as handle:
        pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)