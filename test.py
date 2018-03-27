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
from multiprocessing import Pool, Manager
import os
from functools import partial






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



def process_ont(el, stored_d):
    vec1 = phsim.PhraseVector(el)
    list1, list2, list3= [], [], []
    #print('Finding matches for {}'.format(vec1.phrase))
    for line in df2:
        line_obj= phsim.PhraseVector(line)
        list1.append(vec1.CombinedSimilarity(line_obj))
        #list2.append(vec1.CosineSimilarity(line_obj))
        #list3.append(vec1.WordNetSimilarity(line_obj))
    
    list1 = pd.DataFrame(list1, columns= ['sim_score']).sort_values(by ='sim_score', ascending=False).head(5)
    #list2 = pd.DataFrame(list2, columns= ['sim_score']).sort_values(by ='sim_score', ascending=False).head(5)
    #list3 = pd.DataFrame(list3, columns= ['sim_score']).sort_values(by ='sim_score', ascending=False).head(5)
    
    list1['name'] = [df2[x]  for x in list1.index]
    #list2['name'] = [df2[x]  for x in list2.index]
    #list3['name'] = [df2[x]  for x in list3.index]
    stored_d[el] = list1
    
def update(*a):
    pbar.update()
if __name__ == '__main__':   
    with Manager() as manager:
        stored_d = manager.dict()
        while True:
            try:
                sel = input('Do you want to use word vectors from google or glove? \n')
                MODEL = phsim.PhraseVector.LoadModel(sel)
                break
            except:
                print('invalid input!')
        pbar = tqdm.tqdm(total = 100, ascii=True)  
        pool = Pool(os.cpu_count() - 1)     
        for i in range(pbar.total):                     # Create a multiprocessing Pool
            pool.apply_async(process_ont, args=(df1[i], stored_d,), callback = update)
        pool.close()
        pool.join()
        pbar.close()
        print(stored_d)
        handle = open('ont_data.pickle', 'wb')
        #stored_d only creates a proxy to the real dict!
        pickle.dump(dict(stored_d), handle, pickle.HIGHEST_PROTOCOL)
        handle.close()
        handle = open('ont_data.pickle', 'rb')
        data=pickle.load(handle)
        print(data)
        #handle.close()