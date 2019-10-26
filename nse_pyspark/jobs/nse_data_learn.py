import os
import csv
import pandas as pd
import numpy as np
import collections
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.functions import ltrim,lower,explode,split,expr, col, column,lit,pow,round,bround,corr,count,mean, stddev_pop, min, max,monotonically_increasing_id
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
from pyspark.sql.functions import var_pop, stddev_pop
from pyspark.sql.functions import var_samp, stddev_samp,corr, covar_pop, covar_samp
from pyspark.sql.window import Window
from pyspark.sql.functions import desc

import sys
sys.path
sys.path.append('/Users/kkartikgoel/dev/Python_projects/nse_pyspark')
#create spark session
#del os.environ["PYSPARK_SUBMIT_ARGS"]
spark = SparkSession.builder.appName("NSE").getOrCreate()
#spark.sparkContext.getConf().getAll()
#spark.sparkContext._conf.getAll()
#spark.sparkContext
#spark.conf
#SparkFiles.getRootDirectory()
# read file
#os.listdir("/Users/kkartikgoel/Downloads/Dev practise resources/sec_bhavdata_full.csv")
#print(os.path.realpath(__file__))
#print(os.path.abspath(__file__))

df =  spark.read.csv("hdfs://localhost:9001/lake/files/rundate=20190915/sec_bhavdata_full.csv",header=True,inferSchema=True)
df.printSchema()
df.show(3)
df.columns
header = df.first()
print header
header.show()

[str.strip(column) for column in df.columns]
df.withColumnRenamed(" DATE1",str.strip(" DATE1")).columns
df_strip_spaces = df.toDF(*map(str.strip,df.columns))

df_strip_spaces.show(3)
df_strip_spaces.columns

df_strip_spaces.select(ltrim(df_strip_spaces["SYMBOL"])).show()
df_strip_spaces.select(lower(df_strip_spaces["SYMBOL"])).show()
df_strip_spaces.select(upper(df_strip_spaces["SYMBOL"])).show()
df_strip_spaces.select(lpad(df_strip_spaces["SYMBOL"],20,'0')).show()
df_strip_spaces.DATE1
df_strip_spaces.select(to_date(df_strip_spaces.DATE1),'dd-mmm-yyyy').show()


df_strip_spaces.select(col('SYMBOL')).show(3)
#filtering,selection
df_strip_spaces.select('SYMBOL').show()
df_strip_spaces.select(col('SYMBOL')).show()

df.where(col("SYMBOL").startswith("KOTAK")).show()
df.where("SYMBOL like '%BANK%'").show()
df_strip_spaces.where(col('SYMBOL').like('KOTAKBANK') | col('SYMBOL').like("%YES%")) \
                .select('SYMBOL','DATE1','OPEN_PRICE','CLOSE_PRICE').show(3)

df_strip_spaces.orderBy(to_date("DATE1")).show(3)
df_strip_spaces.orderBy("OPEN_PRICE").where("SERIES like '%EQ'").show(3)

#df.filter(df.SYMBOL.startswith("KOTAK")).show()
#df.filter(col("SYMBOL").startswith("KOTAKBANK")).show()
#df.columns.str.replace(' ','')
df.select(regexp_replace("OPEN_PRICE"," ","")).show(5)
df.where(col("SERIES").like("%EQ")).orderBy(desc(" OPEN_PRICE")).describe().show()
df_strip_spaces.where(col(" SERIES").like("%EQ")).orderBy(to_date(col("DATE1"))).

#counts aggregations
df_strip_spaces.count()
df_strip_spaces.select(count("SERIES")).show()
df_strip_spaces.select(countDistinct("SERIES")).show()
df_strip_spaces.select(approx_count_distinct("SERIES",.1)).show()
df_strip_spaces.select(first("SERIES"),last("SERIES"),min("SERIES"),max("SERIES"),sum("OPEN_PRICE"),sumDistinct("OPEN_PRICE")).show()
df_strip_spaces.select(mean("OPEN_PRICE")).show()
df_strip_spaces.select(avg("OPEN_PRICE")).show()
df_strip_spaces.select(("SERIES"))
df_strip_spaces.groupBy("SERIES","SYMBOL").count().show()
df_strip_spaces.where("SYMBOL like '%BANK%'").groupBy("SERIES").avg().show()
df_strip_spaces.select(avg("OPEN_PRICE"))
df_strip_spaces.select(var_pop("OPEN_PRICE"),stddev_pop("OPEN_PRICE")).show()
df.select(covar_pop("OPEN PRICE","CLOSE PRICE"),corr("OPEN PRICE","CLOSE PRICE")).show()

windowSpec = Window.partitionBy("SYMBOL",to_date("DATE1")).orderBy(to_date("DATE1")).rowsBetween(Window.unboundedPreceding, Window.currentRow)
win = sum(col("OPEN PRICE")).over(windowSpec)
df.select("DATE","OPEN PRICE",win.alias("d")).orderBy(to_date("DATE")).show()
win1 = dense_rank().over(windowSpec)
df.select(rank().over(windowSpec)).show()
