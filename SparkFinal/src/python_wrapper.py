import os,sys
from os.path import abspath
#import csv
#import pandas as pd
#import numpy as np
from subprocess import Popen,PIPE
import logging

def env_vars():
    #envdict = {}
    for key,val in  os.environ.items():
        print (key + "=" + val)
    return os.environ

#os.system("env")

#class log():
def log():
    logdir = os.getcwd()
    logfile = (os.path.basename(__file__)).replace(".py",".log")
    print(logdir,logfile)
    logging.basicConfig(filename=logdir + '/' + logfile, \
                        format='%(asctime)s %(message)s', \
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    return logger

def command():
    cmd = "spark-submit --verbose sparkprod.py > python_spark.log"
    proc = Popen(cmd,shell=True,stdout=PIPE,stderr=PIPE)
    stdout,stderr = proc.communicate()
    print( stdout,stderr )
    if proc.returncode == 0: #proc.communicate()
        return "Spark job finished successfully"
    else:
        return "Spark job failed."

class python_file_handler():

    def check_zerobyte(self,partition,filename):
        proc = Popen("hdfs dfs -test -z /lake/files/%s/%s" % (partition, filename),shell=True,stdout=PIPE,stderr=PIPE)
        proc.communicate()
        return (proc.returncode)

    def read_hive():
        cmd = "beeline" #-u jdbc:hive2://localhost:9000\ -n <yourname> -p <yourpassword> --incremental=true**
        proc = Popen(cmd,shell=True,stdout=PIPE,stderr=PIPE)
        proc.communicate()

    def copy_hdfs(self,partition,filename):
        proc = Popen("hdfs dfs -put resources/%s /lake/files/%s/%s" % (filename,partition, filename),shell=True,stdout=PIPE,stderr=PIPE)
        proc.communicate()
        return (proc.communicate(),proc.returncode)



if __name__ == '__main__':
    #python_file_handler().copy_hdfs("20191001","sec_bhavdata_full_20191001.csv")
    logger = log()
    logger.info("Starting Main Wrapper")
    logger.info("Setting Environment Variables : shell script")
    logger.info("Environment Variables : %s " % env_vars())
    #print(os.path.realpath(__file__))
    logger.info("Readin and Initializing Conifgs and Properties")
    logger.info("Making directories for HDFS Files")
    logger.info("Submitting transformation Spark Job " )
    logger.info(command())
    logger.info("Running Metadata Refresh")

