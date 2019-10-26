import os
import json
import sys
import csv
from datetime import datetime

class calc():
    def __init__(self):
        pass
    def sum_func(self,*args):
        num_list = [ i for i in args]
        print sum(num_list)
    def mul_func(self,*args):
        ans = 1
        for i in args:
            ans = ans * i
        print ans

def cvt_str(str_time):
    return datetime.strptime(str_time,"%Y%m%d")

def cvt_time(time):
    timeobj = cvt_str("20190908")
    return timeobj.strftime("%Y%m%d %H:%M:%S")

def read_json():
    with open("2010-12-01.csv") as f:
        #data = f.read()
        dataline = f.readline()
        #datalines = f.readlines()
        csvreader = csv.reader(f)
        print dataline,csvreader.next()
    with open("etl_config.json",'r+') as f1:
        jsondict = json.load(f1)
        print jsondict
        jsondict["steps_per_floor"] =25
        jsondict1 = json.dumps(jsondict)
        print jsondict1
        open("etl_config.json",'w').write(jsondict1)

    with open("etl_config.json",'w') as f2:
        jsondict = {"steps_per_floor":29}
        json.dump(jsondict,f2)
    f2.close()




def main():
    read_json()
    calc().sum_func(2,3,4,5)
    calc().mul_func(2,3,4,5)
    print cvt_str("20190909")
    print cvt_time("")



if __name__ == '__main__':
    main()
