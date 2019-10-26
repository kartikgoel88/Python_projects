import os
import csv
import pandas as pd
import numpy as np
import collections
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.functions import explode,split,expr, col, column,lit,pow,round,bround,corr,count,mean, stddev_pop, min, max,monotonically_increasing_id
from pyspark.sql.types import StructField, StructType, StringType, LongType
import pyspark.sql.functions as fspark
from pyspark.sql.functions import trim
from pyspark.sql.functions import desc
from pyspark.sql.functions import regexp_replace
from pyspark.sql.functions import *
import collections
import string
from pyspark import SparkFiles
from pyspark.sql import Row
from pyspark.sql.functions import col, concat_ws, lit
import sys
sys.path
sys.path.append('/Users/kkartikgoel/dev/Python_projects/nse_pyspark')
from dependencies.spark import *

def main():
    spark = start_spark(
            app_name='my_etl_job')
            #files=['configs/etl_config.json'])

    #etl pipeline
    data = extract_data(spark)
    data
    data_transformed = transform_data(data)
    load_data(data_transformed)
    spark.stop()
    return None


def extract_data(spark):
    df = (spark.read
    .csv("/Users/kkartikgoel/dev/Python_projects/11-11-2018-TO-08-02-2019KOTAKBANKALLN.csv",header=True,inferSchema=True))
    return df


def transform_data(df):
    df_transformed = (df.where(col("SYMBOL").like('%EQ')).select("SYMBOL","DATE","OPEN PRICE","CLOSE PRICE","TOTAL TRADED QUANTITY"))
    return df_transformed
#.where(col(" SERIES").like('%EQ') & col("SYMBOL").like('%KOTAKBANK'))
def load_data(df):
    (df.coalesce(1)
    .write
    .csv('loaded_data', mode='overwrite', header=True))
    return None


# entry point for PySpark ETL application
if __name__ == '__main__':
    main()
