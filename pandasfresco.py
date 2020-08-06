import pandas as pd
import numpy as np
import os,math


pi = math.pi
np.arccos(2/(np.sqrt(3)*np.sqrt(2)))*(180/pi) # returns 18.434948822922017

# Defining the three dataframes indicating the gold, silver, and bronze medal counts
# of different countries
gold = pd.DataFrame({'Country': ['USA', 'France', 'Russia'],
                         'Medals': [15, 13, 9]}
                    )
silver = pd.DataFrame({'Country': ['USA', 'Germany', 'Russia'],
                        'Medals': [29, 20, 16]}
                    )
bronze = pd.DataFrame({'Country': ['France', 'USA', 'UK'],
                        'Medals': [40, 28, 27]}
                    )

gold.add(silver,axis=1)
print(pd.concat([gold,silver,bronze]).groupby(['Country']).sum(numeric_only=float).astype(float).sort_values(['Medals'],ascending=False))

a = np.array([[4, 3, 1], [5, 7, 0], [9, 9, 3], [8, 2, 4]])
m = 0
n = 2
a[m]
a[n]
# Write your code for swapping here
b = []
c = []
b = a[m]
c= a[n]
print(b)
print(c)
a[m] = c
a[n] = b

# Print the array after swapping
print(a)

list = [176.2,158.4,167.6,156.2,161.4]
heights_A = pd.Series([176.2,158.4,167.6,156.2,161.4],index=['s1', 's2', 's3', 's4','s5'])
heights_A
heights_A.shape
heights_A[1]
heights_A.s2
heights_A[1:4]
heights_A[['s2','s3','s4']]

weights_A = pd.Series([85.1, 90.2, 76.8, 80.4,78.9],index=['s1', 's2', 's3', 's4','s5'])
weights_A
weights_A.dtype

df_A = pd.DataFrame({'Student_height':heights_A,'Student_weight':weights_A})
df_A
df_A['Student_height']

heights_B = pd.Series( 25 * np.random.randn(5) + 170,index=['s1', 's2', 's3', 's4','s5'])
weights_B = pd.Series( 12 * np.random.randn(5) + 75,index=['s1', 's2', 's3', 's4','s5'])
heights_B
weights_B
df_B = pd.DataFrame({'Student_height':heights_B,'Student_weight':weights_B})
df_B
 #column
df_B.Student_height
df_B['Student_height']
 #row
df_B.iloc[[1,2]]
df_B.iloc[1]
#slicing
df_B.index.str.endswith(('1','4'))
df_B.loc[lambda df: df.index.str.endswith(('1','4'))]


#panel
data = {'classA': df_A,'classB': df_B}
p = pd.Panel(data)
p

#write and reads to files
df_A.to_csv('data_fresco_excercise.csv',index=False)
df_A.to_csv('data_fresco_excercise.csv',index=True)
os.path.exists('data_fresco_excercise.csv')
df_A2 = pd.read_csv('data_fresco_excercise.csv')
df_A2 = pd.read_csv('2010-12-01.csv')

df_A3 = pd.read_csv('data_fresco_excercise.csv',index_col=0)
df_A4 = pd.read_csv('data_fresco_excercise.csv',header=None)
df_A5 = pd.read_csv('data_fresco_excercise.csv',header=None,skiprows=1)
df_A2['Quantity'].mean()
df_A2.loc[pd.isnull(df_A2['Quantity'])]
pd.isnull(df_A2['Quantity'])
df_A2[df_A2.isnull().sum(axis=1) > 5]
(df_A2.isnull().sum() /len(df_A2.index))*100
df_A2.index
df_A2.shape
df_A3
df_A3.index
df_A5

#dates
dates = pd.date_range('1-Sep-2017','15-Sep-2017')
dates
datelist = ['14-Sep-2017', '16-Sep-2017']
search_dates = pd.to_datetime(datelist)
search_dates[search_dates.isin(dates)]

#multi index
arraylist = [['classA']*5 + ['classB']*5, ['s1', 's2', 's3','s4', 's5']*2]
mi_index = pd.MultiIndex.from_tuples(arraylist)
arraylist
mi_index.levels

# null values
df_A3.loc['s3'] = np.nan
df_A3.loc['s5','Student_weight'] = np.nan
df_A4 = df_A3.dropna()
df_A4 = df_A3.dropna(subset=['Student_height'])
df_A4.index

#filtering by contents
filter_cond1 = (df_A['Student_height'] > 160.0)  & (df_A['Student_weight']  < 80.0)
df_A[filter_cond1]
df_A.query('Student_height > 160.0')

#filtering by labels
df_A.filter(items=['Student_height'],axis=1)
df_A.filter(regex='5',axis = 0)
df_A[df_A.index.str.endswith('5')]

#column add and grouping
df_A['Gender'] = ['M', 'F', 'M', 'M', 'F']
df_A
df_A.groupby('Gender').mean()
df_A.groupby('Gender').min()
df_A.groupby('Gender').groups
g = df_A.groupby(df_A.index.str.len())
g.filter(lambda x: len(x) > 1)

# adding a row to a dataframe to create a new dataframe
s= pd.Series([165.4, 82.7, 'F'],index= ['Student_height', 'Student_weight', 'Gender'],name='s6')
df_AA = df_A.append(s)
df_AA

# concat 2 df's in a new df
df_B.index = [ 's6', 's7', 's8', 's9', 's10']
df_B['Gender'] = ['F', 'M', 'F', 'F', 'M']
df = pd.concat([df_A,df_B])

# merging and joining 2 dataframes
nameid = pd.Series(range(101, 111))
name = pd.Series(['person' + str(i) for i in range(1, 11)])
master = pd.DataFrame({'nameid':nameid,'name':name})
transaction = pd.DataFrame({'nameid':[108, 108, 108,103], 'product':['iPhone', 'Nokia', 'Micromax', 'Vivo']})
name,nameid
master
transaction
master.merge(transaction,on='nameid')



df = pd.DataFrame({'A':[34, 78, 54], 'B':[12, 67, 43]}, index=['r1', 'r2', 'r3'])
df[:2]
df.iloc[:2]
d = pd.date_range('11-Sep-2017', '17-Sep-2017', freq='2D')
d + pd.Timedelta('1 days 2 hours')

##- test

import pytest
import nbformat
from hashlib import md5
import pandas as pd



class TestJupyter:
    @pytest.fixture(autouse=True)
    def get_notebook(self):
        with open('Users/kkartikgoel/Superhero.ipynb') as f:
            nb = nbformat.read(f, as_version=4)
        self.nb = nb


    def test_thor(self):
        output = self.nb.cells[6].outputs[0].data['text/plain']
        output = output.encode('utf-8')
        assert md5(output).hexdigest()== 'afbc48a7ca8d716f9efa7cc993316668'

    def test_male(self):
        output = self.nb.cells[8].outputs[0].data['text/plain']
        output = output.encode('utf-8')
        assert md5(output).hexdigest()== '451d13a5be2581a451c2284dcecddd4e'

    def test_mt(self):
        output = self.nb.cells[10].outputs[0].data['text/plain']
        output = output.encode('utf-8')
        assert md5(output).hexdigest()== 'c7c8f9b16ebc7282887baeca67236cbe'

    def test_tm(self):
        output = self.nb.cells[12].outputs[0].data['text/plain']
        output = output.encode('utf-8')
        assert md5(output).hexdigest()== '676813538852c7111802c95f5ca99e41'


#--pytest


import inspect
import re
import unittest
import math
import pytest

class InsufficientException(Exception):
    pass

class MobileInventory():

    def __init__(self,inventory={}):
        #mi = MobileInventory()
        if not isinstance(inventory,dict):
            raise TypeError("Input inventory must be a dictionary")
        #type1 = set(type(i) for i in inventory.keys())
        #print(set(map(type,inventory)))
        #set(map(type,inventory)) == {str}
        if [i for i in inventory.keys() if not isinstance(i, str)]:
            raise ValueError("Mobile model name must be a string")
        if [i for i in inventory.values() if not isinstance(i,int) ] :
            raise ValueError("No. of mobiles must be a positive integer")
        if [i for i in inventory.values() if i < 0 ] :
            raise ValueError("No. of mobiles must be a positive integer")
        if inventory is not {}:
            self.balance_inventory = inventory
        else:
            self.balance_inventory = {}

    def add_stock(self,new_stock):
        if not isinstance(new_stock,dict):
            raise TypeError("Input inventory must be a dictionary")
        if [i for i in new_stock.keys() if not isinstance(i, str)]:
            raise ValueError("Mobile model name must be a string")
        if [i for i in new_stock.values() if not isinstance(i,int) ] :
            raise ValueError("No. of mobiles must be a positive integer")
        if [i for i in new_stock.values() if i < 0 ] :
            raise ValueError("No. of mobiles must be a positive integer")
        #if new_stock.keys() in self.balance_inventory.keys():
        for k,v in new_stock.items():
            if k in self.balance_inventory.keys():
                self.balance_inventory[k] = v + self.balance_inventory[k]
            else:
                self.balance_inventory[k] = v
        #print("new stock: {0}".format(self.balance_inventory))

    def sell_stock(self,requested_stock):
        if not isinstance(requested_stock,dict):
            raise TypeError("Input inventory must be a dictionary")
        if [i for i in requested_stock.keys() if not isinstance(i, str)]:
            raise ValueError("Mobile model name must be a string")
        if [i for i in requested_stock.values() if not isinstance(i,int) ] :
            raise ValueError("No. of mobiles must be a positive integer")
        if [i for i in requested_stock.values() if i < 0 ]  :
            raise ValueError("No. of mobiles must be a positive integer")
        for k,v in requested_stock.items():
            if k in self.balance_inventory.keys():
                if self.balance_inventory[k] < v:
                    raise InsufficientException("Insufficient Stock")
                else:
                    self.balance_inventory[k] = self.balance_inventory[k] - v
            else:
                raise InsufficientException("No Stock. New Model Request")

        #print("new stock: {0}".format(self.balance_inventory))

class TestingInventoryCreation():
    def test_creating_empty_inventory(self):
        m = MobileInventory()
        assert m.balance_inventory == {}

    def test_creating_specified_inventory(self):
        m = MobileInventory({'iPhone Model X':100, 'Xiaomi Model Y': 1000, 'Nokia Model Z':25})
        assert m.balance_inventory == {'iPhone Model X':100, 'Xiaomi Model Y': 1000, 'Nokia Model Z':25}

    def test_creating_inventory_with_list(self):
        with pytest.raises(TypeError) as e:
            m = MobileInventory(['iPhone Model X', 'Xiaomi Model Y', 'Nokia Model Z'])
        assert "Input inventory must be a dictionary" in str(e)

    def test_creating_inventory_with_numeric_keys(self):
        with pytest.raises(ValueError) as e:
            m = MobileInventory({1:'iPhone Model X', 2:'Xiaomi Model Y', 3:'Nokia Model Z'})
        assert "Mobile model name must be a string" in str(e)

    def test_creating_inventory_with_nonnumeric_values(self):
        with pytest.raises(ValueError) as e:
            m = MobileInventory({'iPhone Model X':'100', 'Xiaomi Model Y': '1000', 'Nokia Model Z':'25'})
        assert "No. of mobiles must be a positive integer" in str(e)

    def test_creating_inventory_with_negative_value(self):
        with pytest.raises(ValueError) as e:
            m = MobileInventory({'iPhone Model X':-45, 'Xiaomi Model Y': 200, 'Nokia Model Z':25})
        assert "No. of mobiles must be a positive integer" in str(e)




class TestInventoryAddStock():


    @classmethod
    def setup_class(cls):
        cls.m =  MobileInventory( {'iPhone Model X':100, 'Xiaomi Model Y': 1000, 'Nokia Model Z':25})


    def test_add_new_stock_as_dict(self):
        self.m.add_stock({'iPhone Model X':50, 'Xiaomi Model Y': 2000, 'Nokia Model A':10})
        assert self.m.balance_inventory == {'iPhone Model X':150, 'Xiaomi Model Y': 3000, 'Nokia Model Z':25, 'Nokia Model A':10}

    def test_add_new_stock_as_list(self):
        with pytest.raises(TypeError) as e:
            self.m.add_stock(['iPhone Model X', 'Xiaomi Model Y', 'Nokia Model Z'])
        assert "Input inventory must be a dictionary" in str(e)

    def test_add_new_stock_with_numeric_keys(self):
        with pytest.raises(ValueError) as e:
            self.m.add_stock({1:'iPhone Model A', 2:'Xiaomi Model B', 3:'Nokia Model C'})
        assert "Mobile model name must be a string" in str(e)

    def test_add_new_stock_with_nonnumeric_values(self):
        with pytest.raises(ValueError) as e:
            self.m.add_stock({'iPhone Model A':'50', 'Xiaomi Model B':'2000', 'Nokia Model C':'25'})
        assert "No. of mobiles must be a positive integer" in str(e)

    def test_add_new_stock_with_float_values(self):
        with pytest.raises(ValueError) as e:
            self.m.add_stock({'iPhone Model A':50.5, 'Xiaomi Model B':2000.3, 'Nokia Model C':25})
        assert "No. of mobiles must be a positive integer" in str(e)


class TestInventorySellStock():


    @classmethod
    def setup_class(cls):
        cls.m =  MobileInventory( {'iPhone Model A':50, 'Xiaomi Model B': 2000, 'Nokia Model C':10, 'Sony Model D':1})


    def test_sell_stock_as_dict(self):
        self.m.sell_stock({'iPhone Model A':2, 'Xiaomi Model B':20, 'Sony Model D':1})
        assert self.m.balance_inventory == {'iPhone Model A':48, 'Xiaomi Model B': 1980, 'Nokia Model C':10, 'Sony Model D':0}

    def test_sell_stock_as_list(self):
        with pytest.raises(TypeError) as e:
            self.m.add_stock(['iPhone Model A', 'Xiaomi Model B', 'Nokia Model C'] )
        assert "Input inventory must be a dictionary" in str(e)

    def test_sell_stock_with_numeric_keys(self):
        with pytest.raises(ValueError) as e:
            self.m.sell_stock({1:'iPhone Model A', 2:'Xiaomi Model B', 3:'Nokia Model C'} )
        assert "Mobile model name must be a string" in str(e)

    def test_sell_stock_with_nonnumeric_values(self):
        with pytest.raises(ValueError) as e:
            self.m.sell_stock({'iPhone Model A':'2', 'Xiaomi Model B':'3', 'Nokia Model C':'4'})
        assert "No. of mobiles must be a positive integer" in str(e)

    def test_sell_stock_with_float_values(self):
        with pytest.raises(ValueError) as e:
            self.m.sell_stock({'iPhone Model A':2.5, 'Xiaomi Model B':3.1, 'Nokia Model C':4})
        assert "No. of mobiles must be a positive integer" in str(e)

    def test_sell_stock_of_nonexisting_model(self):
        with pytest.raises(InsufficientException) as e:
            self.m.sell_stock({'iPhone Model B':2, 'Xiaomi Model B':5} )
        assert "No Stock. New Model Request" in str(e)

    def test_sell_stock_of_insufficient_stock(self):
        with pytest.raises(InsufficientException) as e:
            self.m.sell_stock({'iPhone Model A':2, 'Xiaomi Model B':5, 'Nokia Model C': 15})
        assert "Insufficient Stock" in str(e)



        #m1 = MobileInventory({'iPhone Model X':100, 'Xiaomi Model Y': 0, 'Nokia Model Z':25})
#m1.add_stock({'iPhone Model X':100})
#m1.sell_stock({'iPhone Model X':100})


--numpy

import numpy as np
import io,re
from StringIO import StringIO
from datetime import datetime
fruits = ['apple', 'mango', 'kiwi', 'watermelon', 'pear']
fruits[:-1:]


#Write detecter implementation
def detecter(element):
    def isIn(sequence):
        if element in sequence:
            return True
        else:
            return False
    return isIn

    #Write isIn implementation

#Write closure function implementation for detect30 and detect45
detect30 = detecter(30)
detect45 = detecter(45)
detect30([2])

def factory(n=0):

    def current():
        return n

    def counter():
        return n + 1
    return current,counter

f_current,f_counter = factory(2)
print(f_current())
print(f_counter())
def fresco():
    zenPython = u'''
    The Zen of Python, by Tim Peters

    Beautiful is better than ugly.
    Explicit is better than implicit.
    Simple is better than complex.
    Complex is better than complicated.
    Flat is better than nested.
    Sparse is better than dense.
    Readability counts.
    Special cases aren't special enough to break the rules.
    Although practicality beats purity.
    Errors should never pass silently.
    Unless explicitly silenced.
    In the face of ambiguity, refuse the temptation to guess.
    There should be one-- and preferably only one --obvious way to do it.
    Although that way may not be obvious at first unless you're Dutch.
    Now is better than never.
    Although never is often better than *right* now.
    If the implementation is hard to explain, it's a bad idea.
    If the implementation is easy to explain, it may be a good idea.
    Namespaces are one honking great idea -- let's do more of those!
    '''

    fp = io.StringIO(zenPython)

    zenlines = fp.readlines()
    zenlines =  [line.strip('') for line in zenlines]
    for line in zenlines:
        portions = re.match("--\w+--",line)
        print(portions)
fresco()

import sqlite3
# establishing  a database connection
con = sqlite3.connect('D:\\TEST.db')
# preparing a cursor object
cursor = con.cursor()
# preparing sql statements
sql1 = 'DROP TABLE IF EXISTS EMPLOYEE'

# establishing the connection
con = sqlite3.connect('D:\\TEST.db')
# preparing a cursor object
cursor = con.cursor()
# preparing sql statement
rec = (456789, 'Frodo', 45, 'M', 100000.00)
sql = '''
      INSERT INTO EMPLOYEE VALUES ( ?, ?, ?, ?, ?)
      '''
#!/bin/python3

import sys
import os
import datetime as dt

#Add log function and inner function implementation here
def log(func):
    def inner(*args, **kwdargs):
        str_template = "Accessed the function -'{}' with arguments {} {}".format(func.__name__,args,kwdargs)
        return str_template + "\n" + str(func(*args, **kwdargs))
    return inner

@log
def greet(msg):
    'Greeting Message : ' + msg

greet("hello")

class EmpNameDescriptor:
    def __get__(self, obj, owner):
        return self.__empname
    def __set__(self, obj, value):
        if not isinstance(value, str):
            raise TypeError("'empname' must be a string.")
        self.__empname = value

class EmpIdDescriptor:
    def __get__(self, obj, owner):
        return self.__empid
    def __set__(self, obj, value):
        if hasattr(obj, 'empid'):
            raise ValueError("'empid' is read only attribute")
        if not isinstance(value, int):
            raise TypeError("'empid' must be an integer.")
        self.__empid = value
class Employee:
    empid = EmpIdDescriptor()
    empname = EmpNameDescriptor()
    def __init__(self, emp_id, emp_name):
        self.empid = emp_id
        self.empname = emp_name

property(fget=None, fset=None, fdel=None, doc=None)

class Employee:
    def __init__(self, emp_id, emp_name):
        self.empid = emp_id
        self.empname = emp_name
    @property
    def empid(self):
        return self.__empid
    @empid.setter
    def empid(self, value):
        if not isinstance(value, int):
            raise TypeError("'empid' must be an integer.")
        self.__empid = value

class Celsius:

    def __get__(self, instance, owner):
        return 5 * (instance.fahrenheit - 32) / 9

    def __set__(self, instance, value):
        instance.fahrenheit = 32 + 9 * value / 5


class Temperature:

    celsius = Celsius()

    def __init__(self, initial_f):
        self.fahrenheit = initial_f
# executing sql statement using try ... except blocks
from abc import ABC, abstractmethod
class Shape(ABC):
    @abstractmethod
    def area(self):
        pass
    @abstractmethod
    def perimeter(self):
        pass
try:

    cursor.execute(sql, rec)

    con.commit()

except Exception as e:

    print("Error Message :", str(e))

    con.rollback()



# closing the database connection

con.close()

addr = ['100 NORTH MAIN ROAD',
            '100 BROAD ROAD APT.',
            'SAROJINI DEVI ROAD',
            'BROAD AVENUE ROAD']
pat = ("ROAD","Rd")
[a.replace(*pat) for a in addr]

def timeConversion(s):
    si = datetime.strptime(s,"%I:%M:%S%p")
    print(si.strftime("%H:%M:%S"))

timeConversion("07:05:45PM")

def birthdayCakeCandles(ar):
    cnt = 0
    maxh = max(ar)
    print(maxh)
    for i in ar:
        if i == maxh:
            cnt += 1
    print(cnt)

birthdayCakeCandles([3 ,2 ,1 ,3])

def miniMaxSum(arr):
    sort = sorted(arr)
    print(sort)
    mins = sum(arr[::-2])
    maxs= sum(arr[1::])
    print(mins)
    print(maxs)

miniMaxSum([7 ,69 ,2 ,221 ,8974])

def staircase(n):
    st = '#'
    for i in range(n):
        #print(i)
        st1 = st * (i+1)
        #print(st1)
        st2 = st1.rjust(n,' ')
        print(st2)

staircase(6)

0
#
     #
1
##
   ##
2
###
 ###
3
####
####
4
#####
#####
5
######
######


'#'*0

def plusMinus(arr):
    cntp = 0
    cntn = 0
    cntz = 0
    for i in arr:
        if i > 0:
            cntp = cntp + 1
        elif i < 0:
            cntn = cntn + 1
        elif i == 0:
            cntz = cntz +1
    tot = len(arr)
    a=cntp/float(tot)
    b=cntn/float(tot)
    c=cntz/float(tot)
    print(a)
    print(b)
    print(c)
    return a,b,c
a = [-4 ,3 ,-9, 0, 4, 1]
plusMinus(a)

def compareTriplets(a,b):
    counta=0
    countb=0
    for i in range(len(a)):
        if a[i] > b[i]:
            counta = counta + 1
        elif a[i] < b[i]:
            countb = countb + 1

    return counta,countb

a = [5,6,7]
b=[3,6,10]
compareTriplets(a,b)

fruits_len = [len(x) for x in fruits]
fruits_len
fruits_mp = [x for x in fruits if (x.startswith("m") or x.startswith("p"))]
fruits_mp

#%save sample_script.py
#%more pandasfresco.py
#%hist
#_i1
#_i2
#_i3
#_1
#_2
#_3

n = [5, 10, 15, 20, 25]
x = np.array(n)
x
x.dtype
x.ndim
x.shape
x.size
x.T

n = [[-1, -2, -3, -4], [-2,-4, -6, -8]]
y = np.array(n)
y.dtype
y.shape
y.size
y.all

n = [[[-1,1],[-2,2]],[[-3 ,3], [-4, 4]]]
x1 = np.array(n)
x1.ndim
x1.shape
x1.size
--numpy end
