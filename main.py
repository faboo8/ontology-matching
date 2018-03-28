from phrase_similarity import PhraseVector, preprocessing_doc
import pandas as pd
from gensim import corpora, models
import tqdm
import pickle
from multiprocessing import Pool, Manager
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

document = preprocessing_doc(pd_full)
#document = [[y for x in document for y in x]]
dict1 = corpora.Dictionary(document)
corpus1 = [dict1.doc2bow(text) for text in document]
tf_model = models.TfidfModel(corpus1, id2word=dict1)
d = {dict1.get(id): value for document in tf_model[corpus1] for id, value in document}



def process_ont(el, stored_d, MODEL):
    vec1 = PhraseVector(el, MODEL)
    print(vec1)
    list1, list2, list3= [], [], []
    #print('Finding matches for {}'.format(vec1.phrase))
    for line in df2:
        line_obj= PhraseVector(line, MODEL)
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
    print(list1)
    
def update(*a):
    pbar.update()
    
    
if __name__ == '__main__':  
    parser = argparse.ArgumentParser()
    parser.add_argument('input1', help='word vector: google or glove', type=str)
    parser.add_argument('input2', help='start index (default 0)', default=0,nargs='?', type =int )
    parser.add_argument('input3', help='end index (defaut max)', default=len(df1),nargs='?', type =int )
    args = parser.parse_args()
    with Manager() as manager:
        stored_d = manager.dict()
        
        while True:
            try:
                sel = args.input1
                MODEL = PhraseVector.LoadModel(sel)
                break
            except:
                print('invalid input!')
                sel = input('Do you want to use word vectors from google or glove? \n')
        print('\n')
        print('\n')
    
        
        total = args.input3-args.input2
        pbar = tqdm.tqdm(total = total, ascii=True)  
        pool = Pool(os.cpu_count() - 1)     
        for i in range(args.input2, args.input3):                     # Create a multiprocessing Pool
            pool.apply_async(process_ont, args=(df1[i], stored_d, MODEL,), callback = update)
        pool.close()
        pool.join()
        pbar.close()
        print(stored_d)
        handle = open('ont_data{}_{}.pickle'.format(args.input2, args.input3), 'wb')
        #stored_d only creates a proxy to the real dict!
        pickle.dump(dict(stored_d), handle, pickle.HIGHEST_PROTOCOL)
        handle.close()
        #handle = open('ont_data.pickle', 'rb')
        #data=pickle.load(handle)
        #print(data)
        #handle.close()