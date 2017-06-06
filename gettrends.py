#! /usr/bin/env python
## trumptrends.py



from pytrends.request import TrendReq
import sys
import os
import os.path
import pandas as pd
import numpy as np
import pickle
import traceback
import time
import random
import string
import json
import datetime


from functools import reduce

# import matplotlib
# # Force matplotlib to not use any Xwindows backend.
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt




### set up the API
google_username = None
google_password = None
path = ""






def random_word(length):
  """Return a random word of 'length' letters.
  https://stackoverflow.com/questions/40406458/google-trends-quota-limit-ip-address-changer
  """
  return ''.join(random.choice(string.ascii_letters) for i in range(length))




def df_overlap_pair(df1, df2):
  """Two data frames.
  They overlap by some amount.
  This only takes the first column if there are multiples.
  """
  # use the first column if there are more than one
  cname = df1.columns.values[0]
  print(cname)
  
  last = df1.index[-1]
  print('last index of df1: %s' % last)
  first = df2.index[0]
  print('first index of df2: %s' % first)
  # If they don't overlap, just stack them, and suffer the non matching scales. oh well.
  if first > last:
    print('!!!!!!!!!!!!!!Warning: concatenating non-overlapping responses: %s %s' % (last, first))
    dfout = pd.concat([df1, df2], axis=0)
    return dfout
  # But if they do overlap, do the normal thing
  endlap = df1.loc[first:,cname]
  #endlap = endlap.values
  startlap = df2.loc[:last,cname]
  #startlap = startlap.values
  #print(endlap)
  #print(startlap)
  endsum = endlap.sum(axis=0)
  startsum = startlap.sum(axis=0)
  print('endsum: %f startsum: %f' % (endsum, startsum))
  # not sure if this is a good idea, try this as a hack
  if startsum==0 and endsum==0:
    startfac = 1.0
    endfac = 1.0
  elif startsum==0:
    print('!!!!!!Warning: startsum=%f endsum=%f using inaccurate reconciliation!' % (startsum, endsum))
    #startsum = np.sign(endsum)
    #startfac = float(endsum)/float(startsum)
    startfac = 1.0
    endfac = 1.0
  elif endsum==0:
    print('!!!!!!Warning: startsum=%f endsum=%f using inaccurate reconciliation!' % (startsum, endsum))
    #endsum = np.sign(startsum)
    #endfac = float(startsum)/float(endsum)
    endfac=1.0
    startfac = 1.0
  elif abs(startsum)>=abs(endsum):
    startfac = float(endsum)/float(startsum)
    endfac = 1.0
  else:
    endfac = float(startsum)/float(endsum)
    startfac = 1.0
  # now apply the factors
  print('endfac: %f startfac: %f' % (endfac, startfac))
  df2out = df2.loc[:,cname] * startfac
  df1out = df1.loc[:,cname] * endfac
  dfoutTemp = pd.concat([df1out, df2out], axis=1)
  dfout = pd.DataFrame(dfoutTemp.mean(axis=1, skipna=True))
  dfout.columns=[cname]
  return dfout


def df_overlap_patch(dflist, refdf):
  """dflist is a list of data frames. they are to be stacked vertically.
  They overlap by some amount. This takes only a single column.
  Uses reduce() with the pairwise function.
  """
  dfout = reduce(df_overlap_pair, dflist)
  # now line it up with the reference dataframe, if that exists
  if refdf is not None:
    dfout = pd.concat([refdf, dfout], axis=1).fillna(0)
    dfout = dfout.iloc[:,[1]]
  return dfout
  
#df_overlap_patch([dflap1, dflap2])



def build_my_payload(qlist, timeframe, pytrendobj=None):
  ntries=5
  for i in range(ntries):
    try:
      print('Try number %d: Building payload with qlist "%s" and timeframe "%s"' % (i+1, qlist, timeframe))
      if not pytrendobj:
        custom_useragent = random_word(8)
        print('  Building TrendReq() object from scratch with custom_useragent %s' % custom_useragent)
        pytrendobj = TrendReq(google_username, google_password, custom_useragent=random_word(8))
      # wait some time to keep it from getting blocked
      sleeptime = (2**i)*random.randint(1,10) # exponential random backoff
      print('Sleeping %d seconds exponential random backoff to avoid getting blocked' % sleeptime)
      time.sleep(sleeptime)
      print('  Building payload with qlist "%s" and timeframe "%s"' % (qlist, timeframe))
      pytrendobj.build_payload(kw_list=qlist, timeframe=timeframe)
      return pytrendobj
    except:
      print(traceback.format_exc())
      print("     Failed to build payload, probably couldn't get token, trying again...")
  print("         Failed to build payload after %d tries, giving up")
  return None


# def build_my_payload_from_scratch(qlist, timeframe):
#   pytrendobj = TrendReq(google_username, google_password, custom_useragent=random_word(8))
#   ntries=5
#   for i in range(ntries):
#     try:
#       print('Building payload with qlist "%s" and timeframe "%s"' % (qlist, timeframe))
#       pytrendobj.build_payload(kw_list=qlist, timeframe=timeframe)
#       return True
#     except:
#       print("     Failed to build payload, probably couldn't get token, trying again...")
#       print(traceback.format_exc())
#   print("         Failed to build payload after %d tries, giving up")
#   return False


# Login to Google. Only need to run this once, the rest of requests will use the same session.
#pytrend = TrendReq(google_username, google_password, custom_useragent=random_word(8))
#pytrend = build_my_payload(['donald trump'], timeframe='today 5-y')
#pytrend.build_payload(kw_list=['testquery'], timeframe='today 5-y')
#print(pytrend.interest_over_time())

# loop through everything and create terms
# we want the who terms, the what terms, and then all the combinations of who and what


def get_trend(qlist, timeframe, refdf=None, pytrendobj=None):
  """wrap the trend call
  """
  # check if timeframe is a string or list
  if hasattr(timeframe, 'upper'):
    # if it's a string, make it a single-item list
    timeframe = [timeframe]
  # now we know timeframe is a list
  dflist = []
  for tf in timeframe:
    print('  About to call build_my_payload for sub-timeframe %s of list %s' % (tf, timeframe))
    pytrendobj = build_my_payload(qlist, tf, pytrendobj)
    # if we really can't get the pytrendobj after all that, then we need to stop and wait for another day
    if pytrendobj is None:
      print('XXXXXX Failed to get trend data for qstring "%s" in %d parts: %s' % (qstring, len(timeframe), timeframe))
      print('XXXXXX Probably means we are blocked. Stop and restart after a day or so.')
      sys.exit()
    try:
      trenddf = pytrendobj.interest_over_time()
      dflist.append(trenddf)
    except:
      # do nothing here
      pass
  if len(dflist) >= 1:
    outdf = df_overlap_patch(dflist, refdf)
  elif len(dflist) == 0:
    outdf = None
  return outdf
  
    


def saveframe(df, filename):
  # You can decide here if you want them saved as compressed pickles, or text, or what
  #filename = filename + '.pkl.gz'
  #df.to_pickle(filename, compression='infer')
  filename = filename + '.csv'
  df.to_csv(filename, compression=None)












def main():
  mydir = os.path.dirname(os.path.realpath(__file__))
  
  timefile = os.path.join(mydir, 'timelist.json')
  termfile = os.path.join(mydir, 'termlist.txt')
  
  # this should be something that is more or less constant over time
  refquery = "google" 
  
  # make a new dir for this run, I'm tired of doing it manually
  datadirname = 'trends_' + datetime.datetime.now().isoformat(sep='T', timespec='minutes').replace(':','-')
  datadir = os.path.join(mydir, datadirname)
  os.mkdir(datadir)
  
  # get the timeframe
  timejson = json.load(open(timefile,'r'))
  timeframe = timejson['timeframe']
  print('Loaded timeframe: %s' % (timeframe,))


  # load terms from text file
  with open(termfile, 'r') as termf:
    termlist = termf.read().splitlines()
  nterms = len(termlist)
  nterms_nodup = len(set(termlist))
  
  print('len(termlist): %d len(set(termlist)): %d' % (nterms, nterms_nodup) )
  if nterms != nterms_nodup:
    print('!!!!!!!! Warning: term list contains %d duplicates' % nterms - nterms.nodup)
  #print(termlist)

  
  # first get a reference data frame using a search term that always works
  qstring=refquery
  print('Querying for string "%s" with timeframe %s' % (qstring, timeframe))
  refdf = get_trend([qstring], timeframe)
  if refdf is not None:
    print('*****Got trend data for reference qstring "%s" with shape %s in %d parts: %s' % (qstring, refdf.shape, len(timeframe), timeframe))
  
  else:
    print('XXXXXX Failed to get trend data for qstring "%s" in %d parts: %s' % (qstring, len(timeframe), timeframe))
    print('We have to give up, we needed that to start!')
    sys.exit()

  #print(refdf)
  # save refdf
  filename = os.path.join(datadir, '__reference_trends')
  saveframe(refdf, filename)
  #sys.exit() # for testing ################################################


  ############### For testing purposes only, I put in a single-item termlist here:
  #termlist = ['covfefe']
  
  nterms = len(termlist)
  for n, qstring in enumerate(termlist):
    print('Query %d of %d for string "%s" with timeframe %s' % (n, nterms, qstring, timeframe))
    trenddf = get_trend([qstring], timeframe, refdf)
    #print(trenddf)
    if trenddf is not None:
      filename = os.path.join(datadir, qstring.replace(' ','_'))
      print('*****Got trend data for %d of %d, qstring "%s" with shape %s in %d parts: %s; saving as %s' % (n, nterms, qstring, trenddf.shape, len(timeframe), timeframe, filename))
      print('*****Received time range: %s to %s' % (trenddf.index[0], trenddf.index[-1]))
      saveframe(trenddf, filename)
    else:
      print('XXXXXX Failed to get trend data for %d of %d qstring "%s" in %d parts: %s' % (n, nterms, qstring, len(timeframe), timeframe))
    # for mini-testing, stop here:
    #sys.exit()
  
  #print(trenddf)




if __name__ == "__main__":
  main()
  
