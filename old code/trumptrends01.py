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

from functools import reduce

# import matplotlib
# # Force matplotlib to not use any Xwindows backend.
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt




### set up the API
google_username = None
google_password = None
path = ""
timeframe='today 5-y'
timeframe = ['2016-05-01 2016-11-01', '2016-10-01 2017-04-01', '2017-03-01 2017-05-29']
# note these overlap by one day, for lining up
#timeframe='now 7-d'


# Factor 1: who

who =     {'Trump':['donald trump','trump', 'president trump', 'trump admin', 'melania trump', 'ivanka trump', 'trump tower'],
          'Hillary':['hillary', 'hillary clinton', 'crooked hillary', 'secretary clinton', 'secretary of state hillary clinton'],
          'Obama':['obama', 'barack obama', 'president obama', 'michelle obama', 'senator obama'],
          'Putin':['putin', 'vladimir putin'],
          'None':['']
          } #19

# for testing
# who =     {'Trump':['trump','donald trump'],
#           'Hillary':['hillary', 'hillary clinton']
#           }


# Factor 2: what

what =    {'Health':['health', 'strength', 'healthy', 'ill', 'sick', 'unhealthy', 'pneumonia', 'weak', 'faint', 'faints', 'seizure', 'seizure disorder'],
          'Dementia':['dementia', 'senility', 'senile', 'alzheimers', 'cognitive impairment'],
          'Intelligence':['intelligence', 'intelligent', 'stupid', 'moron', 'idiot', 'imbecile', 'dumb', 'fool'],
          'Campaign':['campaign','rally','event','speech','supporter','supporters'],
          'Misc':['golf', 'suit', 'dinner', 'conference', 'travel', 'limo', 'bus', 'car', 'hand', 'suit', 'pizza', 'burger', 'steak', 'cake', 'book'],
          'Scandal':['scandal', 'crime', 'criminal', 'treason', 'treasonous', 'traitor', 'collusion', 'complicit', 'benghazi', 'private server', 'emails', 'russia', 'putin', 'fraud', 'money laundering', 'bankrupt', 'bankruptcy', 'scam', 'crook', 'crooked', 'nazi', 'white supremacist', 'hate', 'pussy', 'grab em by the pussy', 'protest', 'mob', 'mafia'],
          'Impeach':['impeach','impeachment','emoluments','unfit'],
          'None':['']
          } # 63

# unused , 'neurodegenerative', 'neurodegeneration', 'neurological'

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
  first = df2.index[0]
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
    startsum = np.sign(endsum)
    startfac = float(endsum)/float(startsum)
    endfac = 1.0
  elif endsum==0:
    endsum = np.sign(startsum)
    endfac = float(startsum)/float(endsum)
    startfac = 1.0
  elif abs(startsum)<=abs(endsum):
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
      sleeptime = random.randint(3,13)
      print('Sleeping %d seconds to avoid getting blocked' % sleeptime)
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
  try:
    for tf in timeframe:
      print('  Building payload for sub-timeframe %s of list %s' % (tf, timeframe))
      pytrendobj = build_my_payload(qlist, tf, pytrendobj)
      trenddf = pytrendobj.interest_over_time()
      if trenddf is not None and trenddf is not False:
        dflist.append(trenddf)
  except:
    print('   Unable to get complete results for qlist "%s" in timeframe list  "%s" at item %s' % (qlist, timeframe, tf))
  if len(dflist) == 1 and len(timeframe) == 1:
    outdf = dflist[0]
  elif len(dflist) > 0 and len(dflist) <= len(timeframe):
    outdf = df_overlap_patch(dflist)
  elif len(dflist) == 0 or len(dflist) != len(timeframe):
    outdf = None
  return outdf
  
    

def oneFactorTrends(factordict, timeframe, pytrendobj=None):
  outlist = []
  for i in factordict.keys():
    for qstring in factordict[i]:
      # for each word, get the data frame
      print('Querying for string "%s" in who=%s with timeframe %s' % (qstring, i, timeframe))
      trenddf = get_trend([qstring], timeframe, pytrendobj)
      if trenddf is not None:
        outlist.append(trenddf)
      else:
        pass
  if len(outlist) > 0:
    outdf = pd.concat(outlist, axis=1)
  else:
    outdf = None
  return outdf


# test
# td = get_trend(['donald trump'], timeframe)
# print(td)
# sys.exit()


# whodf = oneFactorTrends(who, timeframe)
# print('Got whodf, columns:')
# print(whodf.columns.values.tolist())
# print('Writing pickle file for whodf')
# pickle.dump(whodf, open('whodf.pickle', 'wb'), protocol=1)

# whatdf = oneFactorTrends(what, timeframe)
# print('Got whatdf, columns:')
# print(whatdf.columns.values.tolist())
# print('Writing pickle file for whatdf')
# pickle.dump(whatdf, open('whatdf.pickle', 'wb'), protocol=1)




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
          print('*****Got trend data for qstring "%s" with shape %s in %d parts: %s' % (qstring, trenddf.shape, len(timeframe), timeframe))
          #print(trenddf)
          #trenddf.plot()
          #plt.show()
          #plt.savefig('testplot.png')
        else:
          print('XXXXXX Failed to get trend data for qstring "%s" in %d parts: %s' % (qstring, len(timeframe), timeframe))
          pass
        #print(trenddf)
        # for mini-testing, stop here:
        sys.exit()
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


print('Writing pickle file for whowhatdfdict')
pickle.dump(whowhatdfdict, open('whowhatdfdict.pickle', 'wb'), protocol=1)


# whodf = pickle.load(open('whodf.pickle', 'rb'))
# whatdf = pickle.load(open('whatdf.pickle', 'rb'))
# whowhatdfdict = pickle.load(open('whowhatdfdict.pickle', 'rb'))
sys.exit()

for i in whowhatdfdict.keys():
  for j in whowhatdfdict[i].keys():
    v=whowhatdfdict[i][j]
    print(("""'%s %s':""" % (i,j)).replace(' ','_'))
    if v is not None:
      print(repr([x.strip() for x in v.columns.values.tolist()]))
    else:
      print(v)
    print(',')

