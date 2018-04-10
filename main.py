from phrase_similarity import PhraseVector, preprocessing_doc
import pandas as pd
from gensim import corpora, models
import tqdm
import pickle
from multiprocessing import Pool, Manager
import argparse
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


#pd_full = pd.concat([df1,df2])

#document = preprocessing_doc(pd_full)
#document = [[y for x in document for y in x]]
#dict1 = corpora.Dictionary(document)
#corpus1 = [dict1.doc2bow(text) for text in document]
#tf_model = models.TfidfModel(corpus1, id2word=dict1)
#d = {dict1.get(id): value for document in tf_model[corpus1] for id, value in document}



def worker(el, q, vec2_list):
    '''processing'''
    vec1 = el
    list1 = []
    print('Finding matches for {}'.format(vec1.phrase))
    for vec_obj in vec2_list:
        list1.append(vec1.CombinedSimilarity(vec_obj))
    
    list1 = pd.DataFrame(list1, columns= ['sim_score']).sort_values(by ='sim_score', ascending=False).head(5)
    
    list1['name'] = [df2[x]  for x in list1.index]
    
    q.put([vec1.phrase,list1])
    return vec1.phrase, list1

def listener(q):
    '''listens for messages on the q, writes to file. '''
    stored_d = {}
    handle = open('ont_data.pickle', 'wb')
    #data=pickle.load(handle)
    while 1:
        m = q.get()
        if m == 'kill':
            break
        stored_d[m[0]] = m[1]
        
    pickle.dump(stored_d, handle, pickle.HIGHEST_PROTOCOL)       
    handle.flush() 
    handle.close()


def main():
      
    #must use Manager queue here, or will not work
    manager = Manager()
    q = manager.Queue()    
    pool = Pool(processes=4, maxtasksperchild=1000)

    #put listener to work first
    watcher = pool.apply_async(listener, (q,))

    #fire off workers
    jobs = []
    for el in vec1_list:
        job = pool.apply_async(worker, (el, q,vec2_list), callback = update)
        jobs.append(job)

    # collect results from the workers through the pool result queue
    for job in jobs: 
        job.get()

    #now we are done, kill the listener
    q.put('kill')
    pool.close()
    pool.join()
    
    
def update(*a):
    pbar.update()
    
 

if __name__ == '__main__':
    pbar = tqdm.tqdm(total = 5, ascii=True)
    MODEL = PhraseVector.LoadModel('google')
    vec1_list = [PhraseVector(el, MODEL) for el in df1][:5]
    vec2_list = [PhraseVector(el, MODEL) for el in df2]  
    MODEL=None
    main()
    pbar.close()

    
