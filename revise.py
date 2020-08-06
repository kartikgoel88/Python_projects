# python data structures
# lists tuple dict set strings nested lists
list_ex=[3,4,1,2,3,3]
dict_ex={'k' : 'v' }
tuple_ex=(1,2,3,3)
set_ex={1,2,3,3}
string_ex="1233"
print("objs: {} {} {} {}".format(list_ex,dict_ex,tuple_ex,set_ex))
# indexing
print("indexing: {} {} {}".format(list_ex[1],tuple_ex[1],string_ex[1]))
#  slicing
print("slicing: {} {} {}".format(list_ex[1:],tuple_ex[1:],string_ex[1:]))
# add item
dict_ex['a'] = 'b'
print("adding: {} {}".format(list_ex.append(10),dict_ex))
# delete item
print("deletion: {} {}".format(list_ex.pop(0),dict_ex.pop('k')))
# merge objs
# sorting
list_ex.sort()
print("sortin: {} ".format(list_ex))
print("objs: {} {} {} {}".format(list_ex,dict_ex,tuple_ex,set_ex))

# you can write to stdout for debugging purposes, e.g.
# print("this is a debug message")

import datetime
import time
#datetime timezones

#d = datetime.datetime.strptime('2020-02-20','%Y-%m-%d')
#datetime.datetime.strftime(d,'%Y-%m-%d')



import numpy as np

X=int(input("Enter:"))
numbers = list(map(int,input().split(' ')))
print(np.mean())


# def log_health(f):
#     def start_time(*args):
#         start = time.time()
#         result = f(*args)
#         end = time.time()
#         print(end)
#         print( result)
#     return start_time
#
# @log_health
# def process(k):
#     print("job has started but beofre that we need to log health details {}".format(k))
#     return True
# process('hi')
# #log_health("k")
--toptal
import matplotlib
import pandas_datareader.data

R4udWToTe_86q8d595We

def solution(A):
    # write your code in Python 3.6
    print(A)
    a_set=set(A)
    print("a_set:".format(a_set))
    min_set=min(a_set)
    print("min_set:".format(min_set))
    max_set=max(a_set)
    b_set=set(range(min_set,max_set))
    print("b_set:".format(b_set))
    c_set=b_set - a_set
    print(c_set)
    if c_set is None:
        val= max(c_set) + 1
        if val == 0:
            return 1
        else:
            return val

    else:
#         return max(c_set)
# if __name__ == "__main__" :
#     print("starting")
#     solution([2])
--toptal end

def solution(list_ex):
    # write your code in Python 3.6
    a_set=set(list_ex)
    min_set=min(a_set)
    max_set=max(a_set)
    b_set=set(range(min_set,max_set))
    print(b_set)
    c_set=b_set - a_set
    print(c_set)
    if c_set is None:
        val= max(c_set) + 1
        if val == 0:
            return 1
        else:
            return val

    else:
        return max(c_set)


print("input:{}".format(list_ex))
print(solution([1,2,3]))
# pandas data structures
# series dataframes
#
import numpy as np
import pandas as pd


ipl17 = pd.DataFrame({'Team': ['MI', 'RPS', 'SRH', 'KKR', 'KXIP', 'DD', 'GL', 'RCB'],
                      'Matches': [14, 14, 14, 14, 14, 14, 14, 14],
                      'Won': [10, 9, 8, 8, 7, 6, 4, 3],
                      'Lost': [4, 5, 5, 6, 7, 8, 10, 10],
                      'Tied': [0, 0, 0, 0, 0, 0, 0, 0],
                      'N/R': [0, 0, 1, 0, 0, 0, 0, 1],
                      'Points': [20, 18, 17, 16, 14, 12, 8, 7],
                      'NRR': [0.784, 0.176, 0.469, 0.641, 0.123, -0.512, -0.412, -1.299],
                      'For': [2407, 2180, 2221, 2329, 2207, 2219, 2406, 1845],
                      'Against': [2242, 2165, 2118, 2300, 2229, 2255, 2472, 2033]},
                     index = range(1,9)
                     )
print(ipl17['Team'])

ipl18 = pd.DataFrame({'Team': ['SRH', 'CSK', 'KKR', 'RR', 'MI', 'RCB', 'KXIP', 'DD'],
                      'Matches': [14, 14, 14, 14, 14, 14, 14, 14],
                      'Won': [9, 9, 8, 7, 6, 6, 6, 5],
                      'Lost': [5, 5, 6, 7, 8, 8, 8, 9],
                      'Tied': [0, 0, 0, 0, 0, 0, 0, 0],
                      'N/R': [0, 0, 0, 0, 0, 0, 0, 0],
                      'Points': [18, 18, 16, 14, 12, 12, 12, 10],
                      'NRR': [0.284, 0.253, -0.070, -0.250, 0.317, 0.129, -0.502, -0.222],
                      'For': [2230, 2488, 2363, 2130, 2380, 2322, 2210, 2297],
                      'Against': [2193, 2433, 2425, 2141, 2282, 2383, 2259, 2304]},
                     index = range(1,9)
                     )

join = pd.merge(ipl17,ipl18,how='outer',on='Team',validate="one_to_one")
print(join)
print("join1:")
join1 = ipl17.join(ipl18.set_index('Team'),how='outer',on='Team',rsuffix="-18")
print(join1)
print("end join1:")

union = pd.concat([ipl17,ipl18])
print(union)
group = union.groupby('Team',as_index=False)
add = group.sum()
print(add)
srt_points = add.sort_values("Points",ascending=False)
print(srt_points)
select1 = srt_points.iloc[0][['Team']]
print("select1: {}".format(select1))
select2 = srt_points.loc[0]['Team']
print(select2)
pivot1 = union.pivot(columns='Team',values='Points').fillna(0)
print(pivot1)
pivot_tab = pd.pivot_table(union,columns='Team',values='Points',aggfunc=np.sum)
print(pivot_tab)
add_dfs= ipl17.set_index('Team') + ipl18.set_index('Team')
print(add_dfs)
add_dfs1= ipl17.set_index('Team').add(ipl18.set_index('Team'),fill_value=0)
print(add_dfs1)

--datetime

from datetime import datetime ,timedelta,tzinfo
from dateutil.parser import parse
import pandas as pd
#hello!
#python datetime objects

import sys
print(sys.version)
datetime.now()
datetime.now().year
datetime.now().month
datetime.now().day
birthday = datetime(1988,3,8)
#difference and timedelta object
diff = datetime.now() - birthday
diff
diff.days
diff.seconds
diff.days / 365
birthday + timedelta(365)
datetime.now().tzinfo
#datetime to string
str(datetime.now())
datetime.now().strftime('%Y-%m-%d %H:%M:%S %w %z %F %D %a %A')
#python string to datetime
datetime.strptime("1988-03-08","%Y-%m-%d")
parse("1988-03-08") # pandas
parse("1988-03-08", dayfirst=True) # pandas
pd.to_datetime("1988-03-08")
pd.to_datetime(["1988-03-08"])
pd.to_datetime(["1988-03-08"] + [None]) #NaT
pd.to_timedelta(['1 hours']) + birthday

#pandas series and timeseries index - selection, duplicate indices


#Date Ranges, Frequencies, and Shifting - pandas
pd.date_range("1988-03-08","2018-03-08",freq='365d')
pd.date_range("1988-03-08",periods = 10,freq='365d',normalize=True)
--datetime end
