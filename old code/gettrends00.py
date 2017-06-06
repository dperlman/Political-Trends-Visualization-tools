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






# def email_me(msg, addr):
#   import smtplib
#   addr, dom = addr.split('@')
#   smtpObj = smtplib.SMTP(dom, 25)
#   smtpObj.ehlo()
# >>> import smtplib
# >>> smtpObj = smtplib.SMTP('smtp.example.com', 587)
# >>> smtpObj.ehlo()
# (250, b'mx.example.com at your service, [216.172.148.131]\nSIZE 35882577\
# n8BITMIME\nSTARTTLS\nENHANCEDSTATUSCODES\nCHUNKING')
# >>> smtpObj.starttls()
# (220, b'2.0.0 Ready to start TLS')
# >>> smtpObj.login('bob@example.com', ' MY_SECRET_PASSWORD')
# (235, b'2.7.0 Accepted')
# >>> smtpObj.sendmail('bob@example.com', 'alice@example.com', 'Subject: So
# long.\nDear Alice, so long and thanks for all the fish. Sincerely, Bob')
# {}
# >>> smtpObj.quit()
# (221, b'2.0.0 closing connection ko10sm23097611pbd.52 - gsmtp')
# 



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


def df_overlap_patch(dflist):
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


def get_trend(qlist, timeframe, pytrendobj=None):
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
  if len(dflist) == 1:
    outdf = dflist[0]
  elif len(dflist) > 1:
    outdf = df_overlap_patch(dflist)
  elif len(dflist) == 0:
    outdf = None
  return outdf
  
    

















# make a new dir for this run, I'm tired of doing it manually
datadir = 'trends_' + datetime.datetime.now().isoformat(sep='T', timespec='minutes').replace(':','-')
#os.mkdir(datadir)

#timeframe=['today 5-y']
timeframe = ['2016-05-01 2016-11-01', '2016-10-01 2017-04-01', '2017-03-01 2017-06-05']  # note these overlap by one day, for lining up
#timeframe = ['now 7-d']

refdf = None



# load terms from json
termdict = json.load(open('termslist.json', 'r'))
# flatten the dict of lists
termlist = [item for sublist in termdict.values() for item in sublist]
print('len(termlist): %d len(set(termlist)): %d' % (len(termlist), len(set(termlist))) )
print('Removing duplicates and sorting into alphabetical order for no particular reason...')
termlistraw = termlist
termlist = sorted(list(set(termlist)))
print(termlist)

with open('termlist.txt', 'w') as termlistfile:
  termlistfile.write('\n'.join(termlist) + '\n')
sys.exit()

# first get a reference data frame using a search term that always works
qstring='google'
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
filename = os.path.join(datadir, '__reference_trends.pkl.gz')
refdf.to_pickle(filename, compression='infer')
#sys.exit() # for testing ################################################


#termlist=['christopher steele dossier']
termlist = ['covfefe']
nterms = len(termlist)
for n, qstring in enumerate(termlist):
  print('Query %d of %d for string "%s" with timeframe %s' % (n, nterms, qstring, timeframe))
  trenddf = get_trend([qstring], timeframe)
  print(trenddf)
  if trenddf is not None:
    filename = os.path.join(datadir, qstring.replace(' ','_') + '.pkl.gz')
    print('*****Got trend data for %d of %d, qstring "%s" with shape %s in %d parts: %s; saving as %s' % (n, nterms, qstring, trenddf.shape, len(timeframe), timeframe, filename))
    print('*****Received time range: %s to %s' % (trenddf.index[0], trenddf.index[-1]))
    trenddf.to_pickle(filename, compression='infer')
  else:
    print('XXXXXX Failed to get trend data for %d of %d qstring "%s" in %d parts: %s' % (n, nterms, qstring, len(timeframe), timeframe))
  # for mini-testing, stop here:
  #sys.exit()
  
#print(trenddf)


