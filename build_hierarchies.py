# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 10:23:05 2018

@author: DE104752
"""
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='pandas')


import pandas as pd
pd.options.mode.chained_assignment = None
from phrase_similarity import PhraseVector

df_client = pd.read_excel('ARIBA/Warennummern Englisch.xlsx')
df1 = df_client[['CN', 'DESCRIP']]

sections = df1[df1['CN'].str.contains('SECTION')]
chapters = df1[df1['CN'].str.contains('CHAPTER')]
items = df1[~df1["CN"].str.contains('(SECTION)|(CHAPTER)')]
items = items[items['CN'].apply(lambda x: len(str(x).replace(' ',''))==6 )]
###select particular row and get index
#_sections.iloc[0].name

    
    
df_unspsc = pd.read_excel('ARIBA/UNSPSC_Auswahl für DBS.xlsx',skiprows=0)
df_unspsc = df_unspsc.iloc[:,0:4]
df_unspsc.columns= ['domain', 'nrel', 'code', 'DESCRIP']
df_unspsc.drop('domain', inplace=True, axis=1)
df_unspsc.drop('nrel', inplace=True, axis=1)
df2 = df_unspsc['DESCRIP']




### structure of unspsc: SEGEMENT * FAMILY * CLASS * COMMODITY ( = 8 digits)

segment = df_unspsc[df_unspsc['code'].apply(lambda x: len(str(x))==2)]
segment.set_index('code')
family = df_unspsc[df_unspsc['code'].apply(lambda x: len(str(x))==4)]
classes = df_unspsc[df_unspsc['code'].apply(lambda x: len(str(x))==6)]
commodities = df_unspsc[df_unspsc['code'].apply(lambda x: len(str(x))==8)]



####TEST####

MODEL = PhraseVector.LoadModel('google')
vec1 = PhraseVector('Chocolate and other food preparations containing cocoa, in blocks, slabs or bars weighing > 2 kg or in liquid, paste, powder, granular or other bulk form, in containers or immediate packings of a content > 2 kg (excl. cocoa powder)', MODEL, flush = True)
#MODEL = None

        
segment['score'] = segment['DESCRIP'].apply(lambda x: vec1.CombinedSimilarity(PhraseVector(x,MODEL)))
max_seg = segment.loc[segment['score'].idxmax()]['code']
print(max_seg)
    
_family = family[family['code'].apply(lambda x: int(str(x)[:2]) ==max_seg)]
_family['score'] = _family['DESCRIP'].apply(lambda x: vec1.CombinedSimilarity(PhraseVector(x,MODEL)))
max_fam =  _family.loc[_family['score'].idxmax()]['code'] 
print(max_fam)  

_commodities = commodities[commodities['code'].apply(lambda x: int(str(x)[:4]) ==max_fam)]
_commodities['score'] = _commodities['DESCRIP'].apply(lambda x: vec1.CombinedSimilarity(PhraseVector(x,MODEL)))
max_com =  _commodities.loc[_commodities['score'].idxmax()]
print(max_com)
