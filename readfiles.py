# read data from excel, text and database

import os
import csv
import pandas as pd
import numpy as np
from pyspark import SparkContext
from pyspark.sql import SparkSession
import pyspark.sql.functions as fspark
import collections
spark = SparkSession.builder.appName('abc').getOrCreate()



listre = [(101, 'Nik D300', 'Nik D300', 'DSLR Camera', 3), (102, 'Can 1300', 'Can 1300', 'DSLR Camera', 5),
          (103, 'gPhone 13S', 'gPhone 13S', 'Mobile', 10), (104, 'Mic canvas', 'Mic canvas', 'Tab', 5),
          (105, 'SnDisk 10T', 'SnDisk 10T', 'Hard Drive', 1)]
len(listre)
var_string = ', '.join('?' * len(listre))
var_string
conn = sqlite3.connect('SAMPLE.db')
#create connection cursor
cursor1= conn.cursor()
#create table ITEMS using the cursor
query_string = "INSERT INTO ITEMS VALUES (%s);" % var_string
cursor.execute(query_string, varlist)
## files to read
os.environ
os.listdir(u'/Users/kkartikgoel/dev/Python_Projects/ml-100k')
os.chdir(u'/Users/kkartikgoel/dev/Python_Projects/ml-100k')
os.path.exists(‎'KCLT.csv')
os.path.exists(‎"u.data")

## using python
# Read the entire file as a single string
with open('KCLT.csv', 'rt') as f:
    #data = f.read()
    csv_reader = csv.DictReader(f)
    for row in csv_reader:
        print(row)
    #lines = f.readlines(10)
    #print(lines)
for line in lines:
    print(line)
    row = line.split(',')
    print(row)
## using pandas
pd.options.display.max_rows = 10
data_pd = pd.read_csv('KCLT.csv',parse_dates=['date'],nrows=5)
print(data_pd)
tot = pd.Series(data_pd.date)
print tot



## numpy
np.arange(20)
np.linspace(20,5,5)
ndar = np.array([[1,2,3],[4,5,6]])
ndar[0]
np.shape(ndar)
ndar.shape = (3,2)
ndar.reshape(3,2)
ndar
ndar.itemsize
ndar.flags
np.empty([3,2], dtype = int)
np.zeros([3,2], dtype = int)
np.asarray((1,2,3))#convert python sequence into ndarray
#slicing
ndar[:2]
ndar[...,1]
ndar[1,...]
ndar[...,1:]
dt = np.dtype?
type(ndar)


#pandas

obj = pd.Series([4, 7, -5, 3])
obj
obj.name
obj.index.name
obj.values
obj.index
obj[obj > 0]
obj * 2
np.exp(obj)
0 in obj
4 in obj
sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
sdata1 = {'AZ': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
pd.Series(sdata)

#dict of arrays,
data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2001, 2002, 2003],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
frame = pd.DataFrame(data)
frame
frame.head()
frame.loc[3]
frame.T
#dict of Series
data1 = {'number' : sdata,'test' : sdata1}
frame2 = pd.DataFrame(data1)
frame2

# Iterate over the lines of the file
#with open('somefile.txt', 'rt') as f:
#    for line in f:
        # process line

# Read the entire file as a single byte string
#with open('somefile.bin', 'rb') as f:
#    data = f.read()

#  Iterating Over Fixed-Sized Records

#from functools import partial

#RECORD_SIZE = 32

#with open('somefile.data', 'rb') as f:
#    records = iter(partial(f.read, RECORD_SIZE), b'')
#    for r in records:
