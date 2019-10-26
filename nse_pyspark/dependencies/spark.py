import __main__

from os import environ, listdir, path
from json import loads

from pyspark import SparkFiles
from pyspark.sql import SparkSession

#from dependencies import logging

#environ.keys()
def start_spark(app_name='my_spark_app', master='local[*]'):
    # detect execution environment
    flag_repl = False if hasattr(__main__, '__file__') else True
    flag_debug = True if 'DEBUG' in environ.keys() else False

    if not (flag_repl or flag_debug):
        # get Spark session factory
        spark_builder = (
                SparkSession
                .builder
                .appName(app_name))
    else:
        # get Spark session factory
        spark_builder = (
                SparkSession
                .builder
                .master(master)
                .appName(app_name))
    spark_sess = spark_builder.getOrCreate()



    return spark_sess#, spark_logger, config_dict

#','.join(list(['1','2']))
