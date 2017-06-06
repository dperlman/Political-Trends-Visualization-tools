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

import statsmodels.formula.api as sm

from  scipy.ndimage.filters import gaussian_filter1d as gaussfilt
from sklearn.decomposition import FastICA, PCA
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import skew




# Key dates:
# keydates = {'TrumpOfficialNom':pd.to_datetime('2016-07-19'), 
#             'HillaryOfficialNom':pd.to_datetime('2016-07-26'),
#             'TrumpAsksRussiaHack':pd.to_datetime('2016-07-27'),
#             'DeplorablesComment-AltRtConf':pd.to_datetime('2016-09-09'),
#             'September11':pd.to_datetime('2016-09-11'),
#             'FirstDebate':pd.to_datetime('2016-09-26'),
#             'TrumpPussyGrab':pd.to_datetime('2016-10-07'), 
#             'SecondDebate':pd.to_datetime('2016-10-09'),
#             'ThirdDebate':pd.to_datetime('2016-10-19'),
#             'PizzaGateFirst':pd.to_datetime('2016-10-30'),
#             'Election':pd.to_datetime('2016-11-08'), 
#             'MerylStreepSpeech':pd.to_datetime('2017-01-09'),
#             'Inauguration':pd.to_datetime('2017-01-20'),
#             'CrazyPressConf':pd.to_datetime('2017-02-15'),
#             'RussiansOvalOffice':pd.to_datetime('2017-05-15'),
#             'CovfefeTweet':pd.to_datetime('2017-05-31')}
keydates = {pd.to_datetime('2016-06-09'):'Obama Endorses Hillary',
            pd.to_datetime('2016-07-19'):'Trump Officially Nominated', 
            pd.to_datetime('2016-07-26'):'Hillary Officially Nominated',
            pd.to_datetime('2016-07-27'):"Trump Asks Russia to Hack Hillary",
            pd.to_datetime('2016-09-09'):"""Hillary's "Deplorables" Comment; Pneumonia diagnosis""",
            pd.to_datetime('2016-09-11'):"September 11; Hillary Stumbles at Event",
            pd.to_datetime('2016-09-26'):"First Presidential Debate",
            pd.to_datetime('2016-10-07'):"Trump's Pussy-Grab Tape Leaked", 
            pd.to_datetime('2016-10-09'):"Second Presidential Debate",
            pd.to_datetime('2016-10-19'):"Third Presidential Debate",
            pd.to_datetime('2016-10-30'):'"Pizzagate" first appears',
            pd.to_datetime('2016-11-08'):'Election Day', 
            pd.to_datetime('2017-01-10'):"Obama's Farewell; Sessions hearing",
            pd.to_datetime('2017-01-20'):"Trump's Inauguration",
            pd.to_datetime('2017-01-29'):"Immigration Ban Protests",
            pd.to_datetime('2017-02-15'):"Trump's Crazy Press Conference",
            pd.to_datetime('2017-05-15'):"Trump Brings Russians into Oval Office",
            pd.to_datetime('2017-05-31'):'Trump Tweets "Covfefe"'}

#            'MerylStreepSpeech':pd.to_datetime('2017-01-09'):',
#            'FBIAnnounceInvDNCHack':pd.to_datetime('2016-07-25'),
# 'TrumpCampStart':pd.to_datetime('2015-06-16'),

# Load all the individual df results in the data dir
datadir = 'data01'
filepattern = os.path.join(datadir, '*.pkl.gz')
filelist = glob.glob(filepattern)
dflistAll = []
dfdict = {}
for f in filelist:
  searchterm = f[:f.index('.')]
  df = pd.read_pickle(f)
  dflistAll.append(df)
  dfdict[searchterm]=df
# now make one master df
bigdf = pd.concat(dflistAll, axis=1)
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

# ax=bigdf.plot(linewidth=0.1, legend=False)
# for xc in keydates.keys():
#     ax.axvline(x=xc, color='k', linestyle='-', linewidth=0.2)
# plt.show()



# fig = plt.figure(figsize=[10,4], dpi=200)
# ax = fig.add_subplot(111)
# bigdf.mean(axis=1).plot(ax=ax, linewidth=1, legend=False)
# for xc in keydates.values():
#     ax.axvline(x=xc, color='k', linestyle='-', linewidth=0.2)
# ax.set_title('Overview: mean of all full data')
# plt.show()



# narrow down the date range for all of them
# def s(df):
#   #print(len(df.index))
#   dfsub = df['2015-01-01':]
#   nrowOrig = len(dfsub.index)
#   #print(nrowOrig)
#   extra = nrowOrig % 4
#   dfsub = dfsub[extra:]
#   nrow = len(dfsub.index)
#   # if you want to aggregate it, do it here
#   #dfsub = agg(dfsub)
#   return dfsub

# def agg(df):
#   # if you want to aggregate it, do it here
#   i = df.index
#   nrow=len(i)
#   # make the new index
#   iBool = (np.arange(nrow) % 4) == 0
#   ni = i[iBool]
#   #print(ni)
#   # make the grouping index, 0 0 0 0 1 1 1 1 2 2 2 2 etc.
#   gbIndex = np.arange(nrow) // 4
#   #print(gbIndex)
#   groupby = df.groupby(gbIndex, axis=0)
#   dfagg = groupby.aggregate(np.mean)
#   dfagg.index = ni
#   #nrownew = len(dfagg.index)
#   #print(nrowNew)
#   return dfagg



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


###############################################
## Number of components to use in ICA        ##
###############################################
ncomp = 13
###############################################


ica = FastICA(n_components=ncomp, whiten=True)
ica.fit(bigdf)
#icafittrans = ica.fit_transform(bigdf)
icafittrans = ica.transform(bigdf)
print(icafittrans.shape)
icafittrans = pd.DataFrame(icafittrans)
icafittrans.index = dfDateIndex

A_ = ica.mixing_  # Get estimated mixing matrix
# nrows: number of search terms
# ncols: number of components (that we chose)
# so each column is a list of that component's contribution to each search term
# so if we sort a column, that will give us the top search terms for that component!
# but let's make it a data frame and label everything correctly for convenience
A_ = pd.DataFrame(A_)
# set the row names
A_.index = dfSearchTermIndex

W_ = ica.components_ # Get estimated unmixing matrix
# nrows: number of components (that we chose) 
# ncols: number of search terms
# so each row is a list of that component's contribution to each search term, in some other sense different than above
# for comparison, as part of trying to understand how this all works, let's take the transpose and then duplicate the above
WT_ = pd.DataFrame(W_.T)
# set the row names
WT_.index = dfSearchTermIndex



# now let's just test this
# for compnum in range(ncomp):
#   compcol = A_[[compnum]]
#   vals = -np.abs(compcol.values).flatten()
#   sortorder = np.argsort(vals)
#   compcol = compcol.iloc[sortorder, :]
# 
#   compcolW = WT_[[compnum]]
#   valsW = -np.abs(compcolW.values).flatten()
#   sortorderW = np.argsort(valsW)
#   compcolW = compcolW.iloc[sortorderW, :]
#     
#   print(compcol.iloc[:20,:])
#   # my interpretation of looking at the results is that the mixing matrix values make more sense
#   print(compcolW.iloc[:20,:])


def format_termlist(termlist, wrap=40, ncol=2):
  wr = textwrap.TextWrapper(width=wrap, subsequent_indent='  ')
  nterms = len(termlist)
  npercol = int(np.ceil(float(nterms)/ncol))
  outlist=[]
  checklist = []
  for i in range(ncol):
    curlist = termlist[i*npercol:(i+1)*npercol]
    curlist = [wr.fill(i) for i in curlist]
    checklist.append(len(curlist))
    curcolumn = '\n'.join(curlist)
    outlist.append(curcolumn)
  assert sum(checklist) == nterms
  return outlist
  

def bestworstwords(compcol, useabs=False, nterms=20):
  ntermseach = int(nterms/2)
  if useabs:
    vals = -np.abs(compcol.values).flatten()
    sortorder = np.argsort(vals)
    compcol = compcol.iloc[sortorder, :]
    cw1 = compcol.index.values.tolist()[0:ntermseach]
    cw2 = compcol.index.values.tolist()[ntermseach:ntermseach*2]
  else:
    vals=-compcol.values.flatten()
    sortorder = np.argsort(vals)
    compcol = compcol.iloc[sortorder, :]
    compcolwords = compcol.index.values.tolist()
    #cw1 = ['Positive associations:'] + compcolwords[0:ntermseach]
    #cw2 = ['Negative associations:'] + list(reversed(compcolwords[-ntermseach:]))
    cw1 = compcolwords[0:ntermseach]
    cw2 = list(reversed(compcolwords[-ntermseach:]))
  
  cw1 = format_termlist(cw1, wrap=40, ncol=1)[0]
  cw2 = format_termlist(cw2, wrap=40, ncol=1)[0]
  return (cw1, cw2)
    



# look at skewness, gives idea of which way peaks are pointing
#print(skew(A_, axis=0))
# doesn't work well!!

# try this
def peakDirection(aseries, median=False):
  if median:
    mn = np.median(aseries)
  else:
    mn = np.mean(aseries)
  # try it this way first
  largest = max(aseries)
  smallest = min(aseries)
  #print('Largest: %f mn: %f smallest: %f' % (largest, mn, smallest))
  ldif = largest-mn
  smdif = mn-smallest
  #print('ldif: %f smdif: %f' % (ldif, smdif))
  pdir = np.sign(ldif-smdif)
  return pdir

# ICA results are arbitrarily scaled.
# For easy human comprehension we'll rectify the peaks
peakdirs = icafittrans.apply(peakDirection, axis=0)
icafittrans = icafittrans.multiply(peakdirs, axis='columns')
# That's enough to make it look right, but we also need to rectify A_
A_ = A_.multiply(peakdirs, axis='columns')

# While we are at it, let's reorder the components from most peaky to least
def peakiness(aseries, median=False):
  if median:
    mn = np.median(aseries)
  else:
    mn = np.mean(aseries)
  # try it this way first
  largest = max(aseries)
  smallest = min(aseries)
  ldif = largest-mn
  smdif = mn-smallest
  pness = (ldif-smdif)/(largest-smallest)
  return pness

peakinesses = icafittrans.apply(peakiness, axis=0)
peakinessSort = np.argsort(-peakinesses)
# now sort the columns in the icafittrans and the A_
icafittrans = icafittrans.iloc[:,peakinessSort]
A_ = A_.iloc[:,peakinessSort]


#autocorrelation_plot(icafittrans.iloc[:,compnum])
#autocorrelation_plot(icafittrans)



nterms = 20
figsize = [8,10]
figscale = np.mean(figsize)
fig = plt.figure(figsize=figsize, dpi=200)
#fig = plt.figure()
fontsize = figscale * (0.38)
fig.suptitle('Independent Components Analysis of Search Trends for %d Political Terms' % bigdf.shape[1])
for compnum in range(ncomp):
  compcol = A_[[compnum]]
  compcolW = WT_[[compnum]]
  
  
  #cw1, cw2 = format_termlist(compwords, wrap=40)
  cw1, cw2 = bestworstwords(compcol, useabs=False, nterms=20)
  
  ax = fig.add_subplot(ncomp, 1, compnum+1)
  icafittrans.iloc[:,compnum].plot(ax=ax, legend=False, sharex=True, yticks=[])
  
  # temporary, for figuring out peaks
#   mn = np.mean(icafittrans.iloc[:,compnum])
#   #med = np.median(icafittrans.iloc[:,compnum])
#   ax.axhline(y=mn, color='red', linestyle='--', linewidth=0.2)
#   #ax.axhline(y=med, color='red', linestyle=':', linewidth=0.2)
#   largest=max(icafittrans.iloc[:,compnum])
#   smallest=min(icafittrans.iloc[:,compnum])
#   ax.axhline(y=smallest, color='blue', linestyle=':', linewidth=0.2)
#   ax.axhline(y=largest, color='blue', linestyle=':', linewidth=0.2)
#   #print('Largest: %f mn: %f smallest: %f' % (largest, mn, smallest))
  
  txbxwid = 0.165
  r1=Rectangle((-2*txbxwid,0),txbxwid,1.0,transform=ax.transAxes, edgecolor='blue', fill=False, zorder=1, clip_on=False)
  r2=Rectangle((-txbxwid,0),txbxwid,1.0,transform=ax.transAxes, edgecolor='blue', fill=False, zorder=1, clip_on=False)
  ax.add_artist(r1)
  ax.add_artist(r2)
  colmargx = 0.01
  colmargy = 0.05
  ybox1 = ax.text(-2*txbxwid+colmargx,1-colmargy,cw1, fontdict=dict(color="black", size=fontsize, rotation=0,ha='left',va='top'), transform=ax.transAxes)
  ybox2 = ax.text(-txbxwid+colmargx,1-colmargy,cw2, fontdict=dict(color="black", size=fontsize, rotation=0,ha='left',va='top'), transform=ax.transAxes)

  
  for n, evt in enumerate(keydates.keys()):
    # for the top one, label these lines
    if compnum == 0:
      trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
      numlabel = str(n+1)
      eventlabel = ax.text(x=evt, y=1.1, s=numlabel, fontdict=dict(color='black', size=fontsize, rotation=0, ha='center',va='bottom'), transform=trans)
      # and since we're adding text at the top, these headers too:
      poslabel = ax.text(-2*txbxwid+colmargx/2,1+colmargy,s='Positively related words', fontdict=dict(color='black', size=fontsize*1.5, rotation=0, ha='left',va='bottom'), transform=ax.transAxes)
      poslabel = ax.text(-txbxwid+colmargx/2,1+colmargy,s='Negatively related words', fontdict=dict(color='black', size=fontsize*1.5, rotation=0, ha='left',va='bottom'), transform=ax.transAxes)
#     if compnum == ncomp-1:
#       ymin=0
#     else:
#       ymin=-0.1
    ymin=0
    linewidth = 0.2
    linecolor = 'black'
    if evt in [pd.to_datetime('2016-11-08'), pd.to_datetime('2017-01-20')]:
      linewidth = 1
      linecolor = 'red'
      #print('special line for %s: wid %s color %s' % (evt, linewidth, linecolor))
      
    ax.axvline(x=evt, color=linecolor, linestyle='-', linewidth=linewidth, zorder=0, clip_on=False, ymin=ymin, ymax=1.0)
    # print this in a legend at the bottom
  
  
  
  # aesthetic adjustments
  ax.get_xaxis().set_tick_params(which='both', direction='in')
  ax.set_xlabel('')

# aesthetic adjustment once for whole thing:
plt.subplots_adjust(bottom=0.11, left=0.25, top=0.935, right=0.98,hspace=0)

# now add event labels at the bottom
# first just manually cover up the year because I can't figure out how to turn it off
# Note that this is using the last ax, which conveniently is the bottom one
# trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
# yeartime = pd.to_datetime('2017-01-01')
# yearY = -0.3
# yearlocation = (yeartime, yearY)
# yearlocation = trans.transform(yearlocation)
# yrwid = 500
# yrheight = 0.1
# #yearrect=Rectangle(yearlocation,txbxwid, 1.0, transform=ax.transAxes, edgecolor='blue', fill=False, zorder=3, clip_on=False)
# yearrect=Rectangle(yearlocation,txbxwid, 1.0, edgecolor='blue', fill=False, zorder=3, clip_on=False)
# ax.add_artist(yearrect)

eventlabels = []
for n, evt in enumerate(keydates.keys()):
  numlabel = str(n+1)
  keyentry = numlabel + ': ' + keydates[evt]
  eventlabels.append(keyentry)
termkeys = format_termlist(eventlabels, ncol=4)

txbxwid = 0.33
colmargx = 0.01
colmargy = 0.0
xorigin = -1*txbxwid

for n, termkey in enumerate(termkeys):
  
  r = Rectangle( ((xorigin + n*txbxwid),-1.1), txbxwid, 0.83, transform=ax.transAxes, edgecolor=None, facecolor='white', fill=True, zorder=5, clip_on=False)
  ax.add_artist(r)
  keytext = ax.text(xorigin+colmargx + n*txbxwid, -0.32-colmargy, termkey, fontdict=dict(color="black", size=2*fontsize, rotation=0,ha='left',va='top'), transform=ax.transAxes, zorder=6)




#plt.show()
fig.savefig('icacomponents.pdf')
sys.exit()



















fig = plt.figure(figsize=[10,4], dpi=200)
ax = fig.add_subplot(111)
icafittrans.plot(ax=ax, legend=False)
for xc in keydates.values():
    ax.axvline(x=xc, color='k', linestyle='-', linewidth=0.2)
ax.set_title('ICA for full data')
plt.show()

# plot lines individual plots
# for i in range(ncomp):
#   icafittrans.iloc[:,i].plot()
#   plt.show()
# sys.exit()



sys.exit()





# Let's do a PCA on all this data, see what it looks like
pca = PCA(n_components=ncomp)
pcafit = pca.fit(bigdf.iloc[:,:])



fig=plt.figure()
ax=fig.add_subplot(111)
ax.plot(pcafit.explained_variance_ratio_)
ax.set_title('PCA Scree Plot for full data')
plt.show()



pcafittrans = pca.fit_transform(bigdf)
print(pcafittrans.shape)
pcafittrans = pd.DataFrame(pcafittrans)
pcafittrans.index = dfDateIndex
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
#plt.show()
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




