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
from matplotlib.dates import MinuteLocator, HourLocator, DayLocator, WeekdayLocator, MonthLocator, YearLocator, AutoDateLocator
from matplotlib.dates import AutoDateFormatter, DateFormatter, IndexDateFormatter
from matplotlib.ticker import NullLocator, FixedLocator, NullFormatter 


import statsmodels.formula.api as sm

from  scipy.ndimage.filters import gaussian_filter1d as gaussfilt
from sklearn.decomposition import FastICA, PCA
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import skew




# Key dates:
keydates = {pd.to_datetime('2016-06-09'):'Obama Endorses Hillary',
            pd.to_datetime('2016-07-19'):'Trump Officially Nominated', 
            pd.to_datetime('2016-07-22'):'First Wikileaks DNC email dump',
            pd.to_datetime('2016-07-26'):'Hillary Officially Nominated',
            pd.to_datetime('2016-07-27'):"Trump Asks Russia to Hack Hillary",
            pd.to_datetime('2016-08-01'):"Trump criticizes Khan parents; says Russia won't invade Ukraine",
            pd.to_datetime('2016-08-17'):"Guccifer tweets paying back Stone",
            pd.to_datetime('2016-09-09'):"""Hillary's "Deplorables" Comment; Pneumonia diagnosis""",
            pd.to_datetime('2016-09-11'):"September 11; Hillary Stumbles at Event",
            pd.to_datetime('2016-09-26'):"First Presidential Debate",
            pd.to_datetime('2016-10-07'):"Trump's Pussy-Grab Tape Leaked", 
            pd.to_datetime('2016-10-09'):"Second Presidential Debate",
            pd.to_datetime('2016-10-19'):"Third Presidential Debate",
            pd.to_datetime('2016-10-28'):"Comey letter re HRC email investigation",
            pd.to_datetime('2016-10-30'):'"Pizzagate" first appears',
            pd.to_datetime('2016-11-06'):'Second batch of Wikileaks DNC emails', 
            pd.to_datetime('2016-11-08'):'Election Day', 
            pd.to_datetime('2016-12-09'):'CIA announces Russia meddled in election', 
            pd.to_datetime('2017-01-10'):"Obama's Farewell; Sessions hearing",
            pd.to_datetime('2017-01-20'):"Trump's Inauguration",
            pd.to_datetime('2017-01-25'):"Trump fumbles Muir interview",
            pd.to_datetime('2017-01-29'):"Immigration Ban Protests",
            pd.to_datetime('2017-02-15'):"Trump's Crazy Press Conference",
            pd.to_datetime('2017-05-02'):"Morning Joe Implies Trump Dementia",
            pd.to_datetime('2017-05-09'):"Trump fires Jim Comey",
            pd.to_datetime('2017-05-15'):"Trump Brings Russians into Oval Office",
            pd.to_datetime('2017-05-31'):'Trump Tweets "Covfefe"'}


# 2017 March 2: Jeff Sessions recuses himself from Russia investigations
# 2017 March 4: Trump accuses Obama of wiretapping him
# 2017 March 21: bunch of things http://www.cbsnews.com/news/latest-donald-trump-news-today-march-21-2017/
# 2017 March 31: bunch of things http://www.cbsnews.com/news/latest-donald-trump-news-today-march-31-2017/

# less important 2016-06-18 


keydatesh= {pd.to_datetime('2017-05-31'):'Trump Tweets "Covfefe"',
            pd.to_datetime('2017-06-03'):'March for Truth'}








###############################################
## Utility functions to use with components  ##
###############################################

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

###############################################
## Utility functions to use for plotting     ##
###############################################

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
  

def bestworstwords(component, useabs=False, nterms=20, wrap=20):
  #print(component.index)
  ntermseach = int(nterms/2)
  if useabs:
    vals = -np.abs(component.values).flatten()
    sortorder = np.argsort(vals)
    component = component[sortorder]
    compwords = component.index.values.tolist()
    cw1 = compwords[0:ntermseach]
    cw2 = compwords[ntermseach:ntermseach*2]
  else:
    vals=-component.values.flatten()
    sortorder = np.argsort(vals)
    component = component[sortorder]
    #print(component)
    compwords = component.index.values.tolist()
    #cw1 = ['Positive associations:'] + compcolwords[0:ntermseach]
    #cw2 = ['Negative associations:'] + list(reversed(compcolwords[-ntermseach:]))
    cw1 = compwords[0:ntermseach]
    cw2 = list(reversed(compwords[-ntermseach:]))
  
  cw1 = [i.replace('_',' ') for i in cw1]
  cw2 = [i.replace('_',' ') for i in cw2]
  cw1 = format_termlist(cw1, wrap=wrap, ncol=1)[0]
  cw2 = format_termlist(cw2, wrap=wrap, ncol=1)[0]
  return (cw1, cw2, component)
    




###############################################
## The big components plotting function      ##
###############################################


def componentplot(fittrans, components, figtitle, show=False, filename='icacomponents.pdf', keydates=None, nterms=20, figsize=(8,10), dpi=200, fontsize=None, rate='daily'): 
  ncomp = len(fittrans.columns)
  figscale = np.mean(figsize)
  fig = plt.figure(figsize=figsize, dpi=dpi)
  if fontsize is None:
    fsbasis = 0.407*figscale
    fontsize = min(fsbasis, fsbasis * 11.0/(ncomp))
    #fontsize = figscale * (0.38) * 11.0/ncomp - 0.1
  keyfsize = 8.175 * figscale/13.5
  fig.suptitle(figtitle)
  print('Creating plot %d components, fontsize %f' % (ncomp, fontsize))
  for compnum in range(ncomp):
    component = components.iloc[compnum,:]
    cw1, cw2, sortedcomponent = bestworstwords(component, useabs=False, nterms=nterms, wrap=37)
  
    ax = fig.add_subplot(ncomp, 1, compnum+1)
    #print(fittrans.index)
    fittrans.iloc[:,compnum].plot(ax=ax, legend=False, sharex=True, yticks=[])
    # change the x-axis labels to what I want
    if rate.lower()=='daily':
      ax.xaxis.set_major_locator(MonthLocator())
      ax.xaxis.set_major_formatter(DateFormatter('%b'))
    elif rate.lower() == 'hourly':
      #ax.xaxis.set_minor_locator(mdates.HourLocator())
      #ax.xaxis.set_minor_locator(FixedLocator([]))
      #ax.xaxis.set_minor_formatter(NullFormatter())
      #ax.xaxis.set_major_locator(FixedLocator([]))
      #ax.xaxis.set_major_locator(mdates.DayLocator())
      #ax.xaxis.set_major_locator(AutoDateLocator())
      #ax.xaxis.set_major_formatter(DateFormatter('%b %d'))
      pass
  
    txbxwid = 0.165
    r1=Rectangle((-2*txbxwid,0),txbxwid,1.0,transform=ax.transAxes, edgecolor='blue', fill=False, zorder=1, clip_on=False)
    r2=Rectangle((-txbxwid,0),txbxwid,1.0,transform=ax.transAxes, edgecolor='blue', fill=False, zorder=1, clip_on=False)
    ax.add_artist(r1)
    ax.add_artist(r2)
    colmargx = 0.01
    colmargy = 0.02
    ybox1 = ax.text(-2*txbxwid+colmargx, 1-colmargy, cw1, fontdict=dict(color="black", size=fontsize, rotation=0,ha='left',va='top', linespacing=0.94), transform=ax.transAxes)
    ybox2 = ax.text(-txbxwid+colmargx, 1-colmargy, cw2, fontdict=dict(color="black", size=fontsize, rotation=0,ha='left',va='top', linespacing=0.94), transform=ax.transAxes)

    # Make word clouds
    from wordcloud import WordCloud
    def dummyRed(*args, **kwargs):
      return 'red'
    def dummyBlue(*args, **kwargs):
      return 'blue'
      
    nwords = len(sortedcomponent.index)
    ncloud = int(nwords/4)
    #print(type(sortedcomponent)) # it's a Series not DataFrame
    topwords = sortedcomponent[0:ncloud]
    topfreqdict = dict(zip(topwords.index.tolist(), topwords.values.tolist()))
    botwords = -sortedcomponent[-ncloud:]
    botfreqdict = dict(zip(botwords.index.tolist(), botwords.values.tolist())) # it doesn't matter that it's in reversed order, gets sorted by wordcloud
    dirname = os.path.dirname(filename)
    topfname = 'topwc_comp%02dof%02d.png' % ((compnum+1), ncomp)
    botfname = 'botwc_comp%02dof%02d.png' % ((compnum+1), ncomp)
    topfname = os.path.join(dirname, topfname)
    botfname = os.path.join(dirname, botfname)
    
    wc = WordCloud(background_color='white', color_func = dummyRed)
    wc.generate_from_frequencies(topfreqdict)
    topwcimage = wc.to_image()
    with open(topfname, 'wb') as out:
      topwcimage.save(out, format='png')
      
    wc = WordCloud(background_color='white', color_func = dummyBlue)
    wc.generate_from_frequencies(botfreqdict)
    botwcimage = wc.to_image()
    with open(botfname, 'wb') as out:
      botwcimage.save(out, format='png')
      
    # This is useful for playing around, not for final output though
    # overlay plot of word component values in order just for checking
    if False:
      ax2 = ax.twinx().twiny()
      sortedcomponent.plot(ax=ax2, color='red', legend=False, yticks=[], xticks=[])
    
    
    if compnum == 0:
      poslabel = ax.text(-2*txbxwid+colmargx/2,1+colmargy,s='Positively related words', fontdict=dict(color='black', size=keyfsize, rotation=0, ha='left',va='bottom'), transform=ax.transAxes)
      neglabel = ax.text(-txbxwid+colmargx/2,1+colmargy,s='Negatively related words', fontdict=dict(color='black', size=keyfsize, rotation=0, ha='left',va='bottom'), transform=ax.transAxes)
    
    if keydates is not None:
      for n, evt in enumerate(keydates.keys()):
        ymin=0
        linewidth = 0.2
        linecolor = 'black'
        if evt in [pd.to_datetime('2016-11-08'), pd.to_datetime('2017-01-20')]:
          linewidth = 1
          linecolor = 'red'
          #print('special line for %s: wid %s color %s' % (evt, linewidth, linecolor))
      
        ax.axvline(x=evt, color=linecolor, linestyle='-', linewidth=linewidth, zorder=0, clip_on=False, ymin=ymin, ymax=1.0)
  
  
  
    # aesthetic adjustments
    ax.get_xaxis().set_tick_params(which='both', direction='in')
    ax.set_xlabel('')

  # aesthetic adjustment once for whole thing:
  subplbot = 0.13
  subpltop = 0.945
  plt.subplots_adjust(bottom=subplbot, left=0.25, top=subpltop, right=0.98,hspace=0)

  # Add labels for the vertical lines and build a key at the bottom.
  # Label the vertical lines at top and bottom. Use these figure coordinates.
  # note ax is still valid, it's the last one, but they all have same x-coords
  trans = transforms.blended_transform_factory(ax.transData, fig.transFigure)
  eventlabels = []
  ybotorigin = subplbot - 0.027
  for n, evt in enumerate(keydates.keys()):
    numlabel = str(n+1)
    aos = 0.0065 * (n % 2)
    eventlabeltop = ax.text(x=evt, y=subpltop + aos, s=numlabel, fontdict=dict(color='black', size=keyfsize, rotation=0, ha='center',va='bottom'), transform=trans)
    eventlabelbot = ax.text(x=evt, y=ybotorigin + aos, s=numlabel, fontdict=dict(color='black', size=keyfsize, rotation=0, ha='center',va='bottom'), transform=trans)
          
    # now make event labels at the bottom
    keyentry = numlabel + ': ' + evt.strftime('%b %d') + ': ' + keydates[evt]
    eventlabels.append(keyentry)
  
  # and print the event label key at the bottom. once, not in loop.
  termkeys = format_termlist(eventlabels, ncol=4)

  txbxwid = 0.24
  colmargx = 0.01
  colmargy = 0.01
  xorigin = 0.0

  for n, termkey in enumerate(termkeys):
    keytext = fig.text(xorigin+colmargx + n*txbxwid, ybotorigin-colmargy, termkey, fontdict=dict(color="black", size=keyfsize, rotation=0,ha='left',va='top', linespacing=1), transform=ax.transAxes, zorder=6)
    #r = Rectangle( ((xorigin + n*txbxwid),-1.15), txbxwid, 0.83, transform=ax.transAxes, edgecolor=None, facecolor='white', fill=True, zorder=5, clip_on=False)
    #ax.add_artist(r)
    

  if show:
    plt.show()
  fig.savefig(filename)






##################################################
## Now let's have some fun with those functions ##
##################################################



# Load all the individual df results in the data dir
#datadir = 'data01'
#datadir = 'hourlydata01'
# filepattern = os.path.join(datadir, '*.pkl.gz')
# filelist = glob.glob(filepattern)
# dflistAll = []
# dfdict = {}
# for f in filelist:
#   searchterm = f[:f.index('.')]
#   df = pd.read_pickle(f)
#   dflistAll.append(df)
#   dfdict[searchterm]=df
# # now make one master df
# bigdf = pd.concat(dflistAll, axis=1)
# # the list of dates, for re-indexing future data frames
# # the list of terms, for re-indexing future data frames
# minmaxscaler = MinMaxScaler(feature_range=(0,100))
# bigdf = minmaxscaler.fit_transform(bigdf)
# bigdf = pd.DataFrame(bigdf)
# bigdf.index = dfDateIndex
# bigdf.columns = dfSearchTermIndex
# 
# bigdf.columns = [c.replace(' ','_') for c in bigdf.columns]
# #print(bigdf.columns)











###############################################
## Number of components to use in ICA        ##
###############################################
ncomp = 10
###############################################


def myICA(bigdf, ncomp, whiten=True):
  ica = FastICA(n_components=ncomp, whiten=True)
  ica.fit(bigdf)
  icafittrans = ica.transform(bigdf)
  icafittrans = pd.DataFrame(icafittrans)
  icafittrans.index = dfDateIndex

  icacomponents = ica.components_
  icacomponents = pd.DataFrame(icacomponents)
  icacomponents.columns = dfSearchTermIndex

  # ICA results are arbitrarily scaled.
  # For easy human comprehension we'll rectify the peaks
  peakdirs = icafittrans.apply(peakDirection, axis=0)
  icafittrans = icafittrans.multiply(peakdirs, axis='columns')
  # That's enough to make it look right, but we also need to rectify components
  icacomponents = icacomponents.multiply(peakdirs, axis='rows')
  
  # While we are at it, let's reorder the components from most peaky to least
  peakinesses = icafittrans.apply(peakiness, axis=0)
  peakinessSort = np.argsort(-peakinesses)
  # now sort the columns in the icafittrans and the A_
  icafittrans = icafittrans.iloc[:,peakinessSort]
  icacomponents = icacomponents.iloc[peakinessSort,:]
  
  return icafittrans, icacomponents

def myPCA(bigdf, ncomp, whiten=True):
  pca = PCA(n_components=ncomp, whiten=True)
  pca.fit(bigdf)
  pcafittrans = pca.transform(bigdf)
  pcafittrans = pd.DataFrame(pcafittrans)
  pcafittrans.index = dfDateIndex

  pcacomponents = pca.components_
  pcacomponents = pd.DataFrame(pcacomponents)
  pcacomponents.columns = dfSearchTermIndex

  return pcafittrans, pcacomponents
  # PCA results are arbitrarily scaled.
  # For easy human comprehension we'll rectify the peaks
  peakdirs = pcafittrans.apply(peakDirection, axis=0)
  pcafittrans = pcafittrans.multiply(peakdirs, axis='columns')
  # That's enough to make it look right, but we also need to rectify components
  pcacomponents = pcacomponents.multiply(peakdirs, axis='rows')
  
  # While we are at it, let's reorder the components from most peaky to least
  peakinesses = pcafittrans.apply(peakiness, axis=0)
  peakinessSort = np.argsort(-peakinesses)
  # now sort the columns in the icafittrans and the A_
  pcafittrans = pcafittrans.iloc[:,peakinessSort]
  pcacomponents = pcacomponents.iloc[peakinessSort,:]
  


#datafile = 'hourlydata01.csv'
datafile = 'dailydata02.csv'
bigdf = pd.read_csv(datafile, index_col=0, parse_dates=True)

print('Master df bigdf has shape: (%d, %d)' % bigdf.shape)
totalterms = bigdf.shape[1]
print('Number of search terms: %d' % totalterms)
dfDateIndex=bigdf.index
dfSearchTermIndex = bigdf.columns


# number of terms to show for each component, for reference
nterms = 28
#figsize = [8,10]
figsize = [12,15]
dpi = 200 # don't really need this for pdf
figtitle = 'Independent Components Analysis of Search Trends for %d Political Terms' % bigdf.shape[1]

# With ICA, the fit of the model depends on the assumed number of components.
# The components generated by the model are different if a different number is requested.
#complist = [8, 9, 10, 11, 12, 13, 14, 15, 16]
#complist=[8]
complist=[13]
#complist=[6]
for i, ncomp in enumerate(complist): 
  icafittrans, icacomponents = myICA(bigdf, ncomp)
  filename = 'components plots/icacomponents%0d.pdf' % ncomp
  #print([i.strftime('%b %d') for i in icafittrans.index])
  componentplot(icafittrans, icacomponents, figtitle, show=False, filename=filename, keydates=keydates, nterms=nterms, figsize=figsize, dpi=dpi, fontsize=None, rate='hourly')

sys.exit()

figtitle = 'Principal Components Analysis of Search Trends for %d Political Terms' % bigdf.shape[1]

# With PCA, adding more components doesn't change the existing ones, so there's no need to do multiple runs.
complist=[16]
for i, ncomp in enumerate(complist): 
  pcafittrans, pcacomponents = myPCA(bigdf, ncomp)
  filename = 'components plots/pcacomponents%0d.pdf' % ncomp
  componentplot(pcafittrans, pcacomponents, figtitle, show=False, filename=filename, keydates=keydates, nterms=nterms, figsize=figsize, dpi=dpi, fontsize=None)
  





sys.exit()













# Let's do a PCA on all this data, see what it looks like
pca = PCA(n_components=ncomp, whiten=True)
pcafit = pca.fit(bigdf.iloc[:,:])
pcafittrans = pca.transform(bigdf)
pcafittrans = pd.DataFrame(pcafittrans)
pcafittrans.index = dfDateIndex
print(pcafittrans.shape)

fig=plt.figure()
ax=fig.add_subplot(111)
ax.plot(pcafit.explained_variance_ratio_)
ax.set_title('PCA Scree Plot for full data')
plt.show()

# for pca, the "mixing" matrix should be this:




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

















