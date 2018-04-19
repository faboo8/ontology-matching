#!/usr/bin/python
# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings(action='ignore', module='pandas')

from phrase_similarity import PhraseVector
import pandas as pd
import tqdm
import pickle
from multiprocessing import Pool, Manager, freeze_support
import os
import time
import argparse

pd.options.mode.chained_assignment = None


##what to do with word in brackets???

df_client = pd.read_excel('ARIBA/Warennummern Englisch.xlsx')
#df1 = df_client['DESCRIP']
df1 = df_client[~df_client["CN"].str.contains('SECTION|CHAPTER')]
df1 = df1[df1['CN'].apply(lambda x: len(str(x).replace(' ',''))==6 )]
df1.set_index('CN', inplace = True)

df1_as_list = ''
for entry in df1:
    df1_as_list += ' '+ entry
    
    
df_unspsc = pd.read_excel('ARIBA/UNSPSC_Auswahl für DBS.xlsx',skiprows=0)
df_unspsc = df_unspsc.iloc[:,0:4]
df_unspsc.columns= ['domain', 'nrel', 'code', 'DESCRIP']
df_unspsc.drop('domain', inplace=True, axis=1)
df_unspsc.drop('nrel', inplace=True, axis=1)

df2 = df_unspsc['DESCRIP']

df2_as_list = ''
for entry in df2:
    df2_as_list += ' ' + entry
    
    
###building hierarchies
segment = df_unspsc[df_unspsc['code'].apply(lambda x: len(str(x))==2)]


family = df_unspsc[df_unspsc['code'].apply(lambda x: len(str(x))==4)]


classes = df_unspsc[df_unspsc['code'].apply(lambda x: len(str(x))==6)]


commodities = df_unspsc[df_unspsc['code'].apply(lambda x: len(str(x))==8)]



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

def worker_test_hierarchical(el,stored_d,segment, family, classes, commodities, index):
    vec1 = el
    
    
    segment['score'] = segment['pv'].apply(lambda x: vec1.CombinedSimilarity(x, weights = [0.8,0.2]))
    list_score_seg = pd.DataFrame(segment).sort_values(by='score',ascending=False).head(2)['code']
    #max_seg = segment.loc[segment['score'].idxmax()]['code']
    #max_seg_second = 
    
    _family = family[family['code'].apply(lambda x: (int(str(x)[:2]) == list_score_seg.iloc[0]) or  (int(str(x)[:2]) == list_score_seg.iloc[1]))]
    _family['score'] = _family['pv'].apply(lambda x: vec1.CombinedSimilarity(x, weights = [0.8,0.2]))
    max_fam =  _family.loc[_family['score'].idxmax()]['code']
    
    
    _classes = classes[classes['code'].apply(lambda x: int(str(x)[:4]) ==max_fam)]
    _classes['score'] = _classes['pv'].apply(lambda x: vec1.CombinedSimilarity(x, weights = [0.8,0.2]))
    max_clas =  _classes.loc[_classes['score'].idxmax()]['code']

    _commodities = commodities[commodities['code'].apply(lambda x: int(str(x)[:6]) ==max_clas)]
    _commodities['score'] = _commodities['pv'].apply(lambda x: vec1.CombinedSimilarity(x, weights = [0.8,0.2]))
    max_com =  _commodities.loc[_commodities['score'].idxmax()]


    stored_d[vec1.phrase] = max_com
    return stored_d
    #print('\n')
    #return list1


    
def update(*a):
    pbar.update()
def log_result(results):
    result_list.append(results)
    
def errorhandler(exc):
    print('Exception:', exc)
    pbar.update()


result_list = []
    

if __name__ == '__main__':  
    freeze_support()
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
            
        vec1_list = [PhraseVector(el[0], MODEL, flush=True) for el in df1.values]
        #vec2_list = [PhraseVector(el, MODEL, flush=True) for el in df2]
        vec2_list_seg = [PhraseVector(el[1], MODEL, flush=True) for el in segment.values]
        vec2_list_fam = [PhraseVector(el[1], MODEL, flush=True) for el in family.values]
        vec2_list_clas = [PhraseVector(el[1], MODEL, flush=True) for el in classes.values]
        vec2_list_com = [PhraseVector(el[1], MODEL, flush=True) for el in commodities.values]

        segment['pv'] =  vec2_list_seg
        family['pv'] =  vec2_list_fam
        classes['pv'] =  vec2_list_clas
        commodities['pv'] =  vec2_list_com
        ##free up memory
        MODEL = None
        
        total = args.end-args.start
        pbar = tqdm.tqdm(total = total, ascii=True)  
        
        pool = Pool(processes=os.cpu_count() - 1, maxtasksperchild=1000)
        
        #for i in range(args.start,args.end):   
        #    pool.apply_async(worker_test, args=(vec1_list[i],stored_d,vec2_list,i,), callback = update)
        for i in range(args.start,args.end):   
             pool.apply_async(worker_test_hierarchical, args=(vec1_list[i],stored_d,segment, family, classes, commodities, i,), 
                             callback = update, error_callback= errorhandler)
                
        pool.close()
        pool.join()
        
        pbar.close()
        
        handle = open('ont_data{}_{}.pickle'.format(args.start, args.end), 'wb')
        
        #stored_d only creates a proxy to the real dict!
        pickle.dump(dict(stored_d), handle, pickle.HIGHEST_PROTOCOL)
        
        handle.close()
        
#handle = open('ont_data.pickle', 'rb')
#data=pickle.load(handle)
#print(data)
#handle.close()
    
