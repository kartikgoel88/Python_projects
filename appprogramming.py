import numpy as np
import io,re
from StringIO import StringIO
from datetime import datetime
import pdb
from os import environ, listdir, path


pdb.set_trace()
environ
.keys()
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

'''
# test1
def linear_equation(a, b):
    while True:
        x = yield
        e = a*(x**2)+b
        print('Expression, '+str(a)+'*x^2 + '+str(b)+', with x being '+str(x)+' equals '+str(e))

'''
# test 2

# Define 'coroutine_decorator' below
def coroutine_decorator(coroutine_func):
    def wrapper(*args, **kwdargs):
        c = coroutine_func(*args, **kwdargs)
        next(c)
        return c
    return wrapper

# Define coroutine 'linear_equation' as specified in previous exercise
@coroutine_decorator
def linear_equation(a, b):
    while True:
        x = yield
        e = a*(x**2)+b
        print('Expression, '+str(a)+'*x^2 + '+str(b)+', with x being '+str(x)+' equals '+str(e))


# test 3

# Define the function 'coroutine_decorator' below
def coroutine_decorator(coroutine_func):
    def wrapper(*args, **kwdargs):
        c = coroutine_func(*args, **kwdargs)
        next(c)
        return c
    return wrapper

# Define the coroutine function 'linear_equation' below
@coroutine_decorator
def linear_equation(a, b):
    while True:
        x = yield
        e = a*(x**2)+b
        print('Expression, '+str(a)+'*x^2 + '+str(b)+', with x being '+str(x)+' equals '+str(e))


# Define the coroutine function 'numberParser' below
@coroutine_decorator
def numberParser():
    equation1 = linear_equation(3, 4)
    equation2 = linear_equation(2, -1)
    # code to send the input number to both the linear equations
    equation1.send(6.0)
    equation2.send(6.0)
    equation1 = yield
    equation2 = yield
def main(x):
    n = numberParser()
    n.send(x)

def writeTo(filename, input_text):
    with open(filename, 'a') as the_file:
        the_file.write(input_text)
