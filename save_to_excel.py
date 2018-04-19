# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 13:27:28 2018

@author: DE104752
"""

import pickle
import xlsxwriter

handle = open('ont_data{}_{}.pickle'.format(0, 1791), 'rb')
s = pickle.load(handle)

workbook = xlsxwriter.Workbook('data.xlsx')
worksheet = workbook.add_worksheet()
row = 0
col = 0

for key in list(s.keys()):
    row += 1
    worksheet.write(row, col, key)
    i=1
    for item in s[key]:
        #print(type(item))
        if type(item) == int:
            worksheet.write(row, col + 1, item)
        if type(item) == str: 
            worksheet.write(row, col + 2, item)
        else:
            pass
        i += 1
        #else:
         #   pass
         
workbook.close()
handle.close()