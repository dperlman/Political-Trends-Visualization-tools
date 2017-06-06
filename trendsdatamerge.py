#! /usr/bin/env python
## trumptrends.py



import sys
import pickle
import traceback
import time
import random
import string
import os.path
import glob
import textwrap

import numpy as np
import pandas as pd
from pandas.plotting import autocorrelation_plot

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, TextArea, HPacker, VPacker, AnnotationBbox
from matplotlib.text import Text
from matplotlib.patches import Rectangle
import matplotlib.transforms as transforms
import matplotlib.dates as mdates

from sklearn.preprocessing import MinMaxScaler


###############################################
## Utility functions to use with components  ##
###############################################




# Load all the individual df results in the data dir
datadir = 'data01'
#datadir = 'hourlydata01'
datadir = 'dailydata02'

#filepattern = os.path.join(datadir, '*.pkl.gz')
filepattern = os.path.join(datadir, '*.csv')
filelist = glob.glob(filepattern)
dflistAll = []
#dfdict = {}
for f in filelist:
  searchterm = f[:f.index('.')]
  df = pd.read_csv(f, index_col=0, parse_dates=True)
  #df = pd.read_pickle(f)
  dflistAll.append(df)
  #dfdict[searchterm]=df
# now make one master df
bigdf = pd.concat(dflistAll, axis=1)
# Note that if they don't all have the same start and end dates, then there will be NaNs

#bigdf = bigdf.fillna(0) # ????
# probably a better idea:
bigdf = bigdf.dropna(axis=0, how='any')


# the list of dates, for re-indexing future data frames
dfDateIndex=bigdf.index
# the list of terms, for re-indexing future data frames
dfSearchTermIndex = bigdf.columns
minmaxscaler = MinMaxScaler(feature_range=(0,100))
bigdf = minmaxscaler.fit_transform(bigdf)
bigdf = pd.DataFrame(bigdf)
bigdf.index = dfDateIndex
bigdf.columns = dfSearchTermIndex

bigdf.columns = [c.replace(' ','_') for c in bigdf.columns]
#print(bigdf.columns)
print('Master df bigdf has shape: (%d, %d)' % bigdf.shape)
totalterms = bigdf.shape[1]
print('Number of search terms: %d' % totalterms)

print(bigdf.iloc[1,:])
print(bigdf.iloc[-1,:])

bigdf.to_csv(os.path.join(datadir, 'alldata.csv'))



















