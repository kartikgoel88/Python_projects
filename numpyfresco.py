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
