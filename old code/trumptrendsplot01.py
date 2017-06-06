#! /usr/bin/env python
## trumptrends.py



from pytrends.request import TrendReq
import sys
import pandas as pd
import numpy as np
import pickle
import traceback
import time
import random
import string

import matplotlib.pyplot as plt
from  scipy.ndimage.filters import gaussian_filter1d as gaussfilt
import statsmodels.formula.api as sm
from sklearn.decomposition import FastICA, PCA



# Key dates:
keydates = {'TrumpCampStart':pd.to_datetime('2015-06-16'), 
            'TrumpOfficialNom':pd.to_datetime('2016-07-19'), 
            'DeplorablesComment-AltRtConf':pd.to_datetime('2016-09-09'),
            'TrumpPussyGrab':pd.to_datetime('2016-10-07'), 
            'PizzaGateFirst':pd.to_datetime('2016-10-30'),
            'Election':pd.to_datetime('2016-11-08'), 
            'Inauguration':pd.to_datetime('2017-01-20'),
            'CrazyPressConf':pd.to_datetime('2017-02-15'),
            'RussiansOvalOffice':pd.to_datetime('2017-05-15')}


# Factor 1: who

who =     {'Trump':['donald trump','trump', 'president trump', 'trump admin', 'melania trump', 'ivanka trump', 'trump tower'],
          'Hillary':['hillary', 'hillary clinton', 'crooked hillary', 'secretary clinton', 'secretary of state hillary clinton'],
          'Obama':['obama', 'barack obama', 'president obama', 'michelle obama', 'senator obama'],
          'Putin':['putin', 'vladimir putin']
          } #19

# for testing
# who =     {'Trump':['trump','donald trump'],
#           'Hillary':['hillary', 'hillary clinton']
#           }


# Factor 2: what

what =    {'Health':['health', 'strength', 'healthy', 'ill', 'sick', 'unhealthy', 'pneumonia', 'weak', 'faint', 'faints', 'seizure', 'seizure disorder'],
          'Dementia':['dementia', 'senility', 'senile', 'alzheimers', 'cognitive impairment', 'neurodegenerative', 'neurodegeneration', 'neurological'],
          'Intelligence':['intelligence', 'intelligent', 'stupid', 'moron', 'idiot', 'imbecile', 'dumb', 'fool'],
          'Misc':['golf', 'suit', 'dinner', 'rally', 'conference', 'speech', 'event', 'supporter', 'supporters', 'travel', 'limo', 'bus', 'car', 'hand', 'suit', 'pizza', 'burger', 'steak', 'cake', 'book'],
          'Crime':['scandal', 'crime', 'criminal', 'treason', 'treasonous', 'traitor', 'collusion', 'complicit', 'benghazi', 'russia', 'fraud', 'money laundering', 'scam', 'crook', 'crooked']
          } # 63


def sr(x):
  x=x.replace(' ','_')
  return x
  
# whodf = pickle.load(open('whodf.pickle', 'rb'))
# whatdf = pickle.load(open('whatdf.pickle', 'rb'))
whodf = pd.read_pickle('whodf.pickle')
whatdf = pd.read_pickle('whatdf.pickle')
#whowhatdfdict = pickle.load(open('whowhatdfdict.pickle', 'rb'))
whowhatdfdict = pd.read_pickle('whowhatdfdict.pickle')


whodf.columns = [c.replace(' ','_') for c in whodf.columns]
whatdf.columns = [c.replace(' ','_') for c in whatdf.columns]
#whodf=whodf.rename(columns=sr)
#whatdf=whatdf.rename(columns=sr)

#print(whodf.columns)
#print(whatdf.columns)

#print(whowhatdfdict.keys())

# narrow down the date range for all of them
def s(df):
  #print(len(df.index))
  dfsub = df['2015-01-01':]
  nrowOrig = len(dfsub.index)
  #print(nrowOrig)
  extra = nrowOrig % 4
  dfsub = dfsub[extra:]
  nrow = len(dfsub.index)
  # if you want to aggregate it, do it here
  #dfsub = agg(dfsub)
  return dfsub

def agg(df):
  # if you want to aggregate it, do it here
  i = df.index
  nrow=len(i)
  # make the new index
  iBool = (np.arange(nrow) % 4) == 0
  ni = i[iBool]
  #print(ni)
  # make the grouping index, 0 0 0 0 1 1 1 1 2 2 2 2 etc.
  gbIndex = np.arange(nrow) // 4
  #print(gbIndex)
  groupby = df.groupby(gbIndex, axis=0)
  dfagg = groupby.aggregate(np.mean)
  dfagg.index = ni
  #nrownew = len(dfagg.index)
  #print(nrowNew)
  return dfagg


# clean them all up
# and also make one master df
whodf = s(whodf)
whatdf = s(whatdf)
bigdf = whodf.copy()
bigdf=pd.concat([bigdf, whatdf], axis=1)
for j in whowhatdfdict.keys():
  for k in whowhatdfdict[j].keys():
    if whowhatdfdict[j][k] is not None:
      whowhatdfdict[j][k] = s(whowhatdfdict[j][k])
      whowhatdfdict[j][k] = whowhatdfdict[j][k].rename(columns=sr)
      bigdf = pd.concat([bigdf, whowhatdfdict[j][k]], axis=1)
    else:
      #whowhatdfdict[j].pop(k)
      pass
      
print('Master df bigdf has shape: (%d, %d)' % bigdf.shape)
print(bigdf.shape)
# ax=bigdf.plot(linewidth=0.1, legend=False)
# for xc in keydates.values():
#     ax.axvline(x=xc, color='k', linestyle='-', linewidth=0.2)
# plt.show()
# ax=bigdf.mean(axis=1).plot(linewidth=1, legend=False)
# for xc in keydates.values():
#     ax.axvline(x=xc, color='k', linestyle='-', linewidth=0.2)
# plt.show()




trump=whodf[['donald_trump','trump','president_trump','trump_admin']]
hillary=whodf[['hillary','hillary_clinton','secretary_clinton','secretary_of_state_hillary_clinton']]
obama=whodf[['obama','barack_obama','president_obama','senator_obama']]

#dementia = whatdf[['dementia','senile','senility','alzheimers','cognitive_impairment','neurodegenerative']]
dementia = whatdf[['dementia','senile','alzheimers']]
stupid = whatdf[['intelligence','intelligent','stupid','moron','idiot','imbecile','dumb','fool']]
health = whatdf[['health','strength','healthy','ill','sick','unhealthy','pneumonia','weak','faint','faints','seizure']]
crime = whatdf[['scandal', 'crime', 'criminal', 'treason', 'treasonous', 'traitor','collusion', 'complicit', 'benghazi', 'russia', 'fraud','money_laundering', 'scam', 'crook', 'crooked']]

trumpdementia = whowhatdfdict['Trump']['Dementia'].drop('trump_neurological',axis=1) # this one has minimal data, adds noise
hillarydementia = whowhatdfdict['Hillary']['Dementia']
#obamadementia = whowhatdfdict['Obama']['Dementia']
trumphealth = whowhatdfdict['Trump']['Health']
trumphealth = trumphealth.select(lambda x: ('melania' not in x.lower()) and ('ivanka' not in x.lower()), axis=1) # these aren't actually about Donald Trump
hillaryhealth = whowhatdfdict['Hillary']['Health']
obamahealth = whowhatdfdict['Obama']['Health']
trumpcrime = whowhatdfdict['Trump']['Crime']
hillarycrime = whowhatdfdict['Hillary']['Crime']
obamacrime = whowhatdfdict['Obama']['Crime']

# print(whowhatdfdict['Obama']['Health'])
# print(whowhatdfdict['Obama']['Intelligence'])
# print(whowhatdfdict['Obama']['Misc'])
# print(whowhatdfdict['Obama']['Crime'])
# print(whowhatdfdict['Obama']['Dementia'])

# print('HillaryDement:  ' + ', '.join(hillarydementia.columns.values.tolist()))
# #print('ObamaDementia:  ' + ', '.join(obamadementia.columns.values.tolist()))
# print('HillaryHealth:  ' + ', '.join(hillaryhealth.columns.values.tolist()))
# print('ObamaHealth  :  ' + ', '.join(obamahealth.columns.values.tolist()))





dtdemMean = trumpdementia.mean(axis=1).reindex_like(trumpdementia)
hcdemMean = hillarydementia.mean(axis=1)
dthelMean = trumphealth.mean(axis=1)
hchelMean = hillaryhealth.mean(axis=1)
bohelMean = obamahealth.mean(axis=1)
dtcriMean = trumpcrime.mean(axis=1)
hccriMean = hillarycrime.mean(axis=1)
bocriMean = obamacrime.mean(axis=1)


# The basic plot of Google Trends data for Trump senility
# print('TrumpDementia:  ' + ', '.join(trumpdementia.columns))
# trumpdementia['Mean'] = trumpdementia.mean(axis=1)
# fig = plt.figure(figsize=[10,4], dpi=200)
# ax = fig.add_subplot(111)
# trumpdementia.plot(linewidth=0.7, ax=ax, title="Google Trends: Trump Senility")
# ml=ax.lines[-1]
# plt.setp(ml, linewidth=2, color='black')
# ax.legend(loc=2,prop={'size':6})
# plt.show()


#print(trumphealth.iloc[-2:])
#print(trumphealth.iloc[-1]>trumphealth.iloc[-2])

# will use this after this one
trumphealthup = trumphealth.copy(deep=False)

# Basic plot of Google Trends for Trump health
# print('TrumpHealth  :  ' + ', '.join(trumphealth.columns.values.tolist()))
# trumphealth['Mean'] = trumphealth.mean(axis=1)
# fig = plt.figure(figsize=[10,4], dpi=200)
# ax = fig.add_subplot(111)
# trumphealth.plot(linewidth=0.7, ax=ax, title="Google Trends: Trump Health")
# ml=ax.lines[-1]
# plt.setp(ml, linewidth=2, color='black')
# ax.legend(loc=2,prop={'size':4}, ncol=2)
# plt.show()

# Repeat that, but with only the ones trending upwards
# Maybe if we do that, the senile plot doesn't look so extreme in comparison

# updist=6
# upcols=(trumphealthup.iloc[-1]>trumphealthup.iloc[-(updist+1)])
# print(upcols)
# print(trumphealthup.shape)
# trumphealthup = trumphealthup.loc[:,upcols.values]
# print(trumphealthup.shape)
# trumphealthup['Mean'] = trumphealthup.mean(axis=1)
# fig = plt.figure(figsize=[10,4], dpi=200)
# ax = fig.add_subplot(111)
# trumphealthup.plot(linewidth=0.7, ax=ax, title="Google Trends increasing recently: Trump Health")
# ml=ax.lines[-1]
# plt.setp(ml, linewidth=2, color='black')
# ax.legend(loc=2,prop={'size':4}, ncol=2)
# plt.show()
# Nope, actually health isn't really trending up at all recently.


# How about some nice ICA with your data
#FastICA(n_components=n_components, whiten=True),
# Compute ICA
# ica = FastICA(n_components=3)
# S_ = ica.fit_transform(X)  # Reconstruct signals
# A_ = ica.mixing_  # Get estimated mixing matrix
ncomp = 9
ica = FastICA(n_components=ncomp, whiten=True)
icafittrans = ica.fit_transform(bigdf)
print(icafittrans.shape)
icafittrans = pd.DataFrame(icafittrans)
icafittrans.index = whodf.index
fig = plt.figure(figsize=[10,4], dpi=200)
ax = fig.add_subplot(111)
icafittrans.plot(ax=ax)
for xc in keydates.values():
    ax.axvline(x=xc, color='k', linestyle='-', linewidth=0.2)
plt.show()

# plot lines individual plots
# for i in range(ncomp):
#   icafittrans.iloc[:,i].plot()
#   plt.show()
# sys.exit()

# Let's do a PCA on all this data, see what it looks like
pca = PCA(n_components=ncomp)
pcafit = pca.fit(bigdf.iloc[:,:])

plt.figure()
plt.plot(pcafit.explained_variance_ratio_)
plt.show()

pcafittrans = pca.fit_transform(bigdf)
print(pcafittrans.shape)
pcafittrans = pd.DataFrame(pcafittrans)
pcafittrans.index = whodf.index
fig = plt.figure(figsize=[10,4], dpi=200)
ax = fig.add_subplot(111)
pcafittrans.plot(ax=ax)
for xc in keydates.values():
    ax.axvline(x=xc, color='k', linestyle='-', linewidth=0.2)
plt.show()


sys.exit()
sig = 4
meansdfSmooth = meansdf.copy()
print('Smoothing means into new data frame')
for column in meansdfSmooth:
  meansdfSmooth[column] = gaussfilt(meansdf[column], sigma=sig)


plt.plot(meansdf['DTHealth'],meansdf['DTDementia'])
plt.show()
plt.plot(meansdfSmooth['DTHealth'],meansdfSmooth['DTDementia'])
plt.show()
meansdfSmooth['DTDemMinusHel'] = meansdfSmooth['DTDementia'] - meansdfSmooth['DTHealth']
meansdfSmooth['DTDemMinusHel'].plot()
plt.show()

ax=meansdf[['DTHealth','DTDementia']].plot()
meansdfSmooth[['DTHealth','DTDementia']].plot(ax=ax)
plt.show()



#meansdfSmooth = pd.DataFrame({'DTDementia':gaussfilt(dtdemMean, sigma=sig), 'HCDementia':gaussfilt(hcdemMean, sigma=sig), 'DTHealth':gaussfilt(dthelMean, sigma=sig), 'HCHealth':gaussfilt(hchelMean, sigma=sig), 'BOHealth':gaussfilt(bohelMean, sigma=sig)})
meansdfSmooth.plot()
plt.show()
#print(meansdf)





sys.exit()














whowhatdict = {}
whowhatdfdict = {}
for i in who.keys():
  print('Cycling through who=%s' % i)
  whowhatdict[i]={}
  whowhatdfdict[i]={}
  for ii in what.keys():
    print('Cycling through what=%s in who=%s' % (ii, i))
    whowhatdict[i][ii]=[]
    for j in who[i]:
      for jj in what[ii]:
        # for each combination, get the data frame
        qstring = j + ' ' + jj
        print('Querying for string "%s" in what=%s who=%s with timeframe %s' % (qstring, ii,i, timeframe))
        trenddf = get_trend([qstring], timeframe)
        if trenddf is not None:
          whowhatdict[i][ii].append(trenddf)
        else:
          pass
      # for testing, stop after one inner loop
      #tempdf = pd.concat(whowhatdict[i][ii], axis=1)
      #print(tempdf)
      #sys.exit()
    if len(whowhatdict[i][ii]) > 0:
      whowhatdf = pd.concat(whowhatdict[i][ii], axis=1)
      whowhatdfdict[i][ii] = whowhatdf
      print('Got whowhatdf, columns:')
      print(whowhatdf.columns.values.tolist())
    else:
      whowhatdfdict[i][ii] = None




