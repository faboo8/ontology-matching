#!/usr/bin/python
# -*- coding: utf-8 -*-

from phrase_similarity import PhraseVector
import pandas as pd
import tqdm
import pickle
from multiprocessing import Pool, Manager, freeze_support
import os
import argparse


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

#document = preprocessing_doc(pd_full)
#document = [[y for x in document for y in x]]
#dict1 = corpora.Dictionary(document)
#corpus1 = [dict1.doc2bow(text) for text in document]
#tf_model = models.TfidfModel(corpus1, id2word=dict1)
#d = {dict1.get(id): value for document in tf_model[corpus1] for id, value in document}



def worker_test(el,stored_d,vec2_list,index):
    vec1 = el
    list1 = []
    #print('Finding matches for {}'.format(vec1.phrase))
    for vec_obj in vec2_list:
        list1.append(vec1.CombinedSimilarity(vec_obj))
        
    
    list1 = pd.DataFrame(list1, columns= ['sim_score']).sort_values(by ='sim_score', ascending=False).head(5)
    
    
    list1['name'] = [df2[x]  for x in list1.index]

    stored_d[index] = list1
    print('\n')
    return list1

    
def update(*a):
    pbar.update()
def log_result(results):
    result_list.append(results)
    
def errorhandler(exc):
    print('Exception:', exc)

result_list = []
    

if __name__ == '__main__':  
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='word vector: google or glove', type=str)
    parser.add_argument('start', help='start index (default 0)', default=0,nargs='?', type =int )
    parser.add_argument('end', help='end index (defaut max)', default=len(df1),nargs='?', type =int )
    args = parser.parse_args()
    
    with Manager() as manager:
        stored_d = manager.dict()
        
        while True:
            try:
                sel = args.model
                MODEL = PhraseVector.LoadModel(sel)
                break
            except:
                print('invalid input!')
                sel = input('Do you want to use word vectors from google or glove? \n')
                MODEL = PhraseVector.LoadModel(sel)
                break
            
        vec1_list = [PhraseVector(el, MODEL) for el in df1]
        vec2_list = [PhraseVector(el, MODEL) for el in df2]
        MODEL = None
        
        total = args.end-args.start
        pbar = tqdm.tqdm(total = total, ascii=True)  
        pool = Pool(processes=os.cpu_count() - 1, maxtasksperchild=1000)
        
        for i in range(args.start,args.end):   
            pool.apply_async(worker_test, args=(vec1_list[i],stored_d,vec2_list,i,), callback = update)
            
        pool.close()
        pool.join()
        pbar.close()
        #print(stored_d)
        
        handle = open('ont_data{}_{}.pickle'.format(args.start, args.end), 'wb')
        
        #stored_d only creates a proxy to the real dict!
        pickle.dump(dict(stored_d), handle, pickle.HIGHEST_PROTOCOL)
        handle.close()
        
#handle = open('ont_data.pickle', 'rb')
#data=pickle.load(handle)
#print(data)
#handle.close()
    
