
import os,sys
from os.path import abspath
import csv
#import pandas as pd
#import numpy as np
from pyspark import SparkFiles,SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from subprocess import Popen,PIPE

sys.path.append("/Users/kkartikgoel/dev/Python_Projects/pyspark-example-project-master")
from dependencies import logging


def start_spark(app,files=[],pyfiles=[]):
## Spark Context ##
    #conf = SparkConf().setAppName("AppSparkContext").set("spark.files","etl_config.json").toDebugString()
    #sc = SparkContext(conf=conf)
    #warehouse_location = abspath('hdfs://localhost:9001/lake/files')

## Spark Session ##
    spark_builder = SparkSession.builder.appName(app).master('local[*]')
    spark_builder.config('spark.files',"SparkFinal/configs/etl_config.json,/usr/local/Cellar/hive/2.1.0/libexec/conf/hive-site.xml")
    #spark_builder.config('spark.logConf','true')
    #spark_builder.config('spark.jars.repositories','/Users/kkartikgoel/dev/spark-2.1.0-bin-hadoop2.7/jars')
    #spark_builder.config('spark.jars.packages','com.databricks:spark-avro_2.10:1.0.0')
    #spark_builder.config('spark.jars.packages','com.databricks:spark-avro_2.10:1.0.0')
    #spark_builder.config('hive.metastore.uris','thrift://localhost:9083')
    spark_sess = spark_builder.enableHiveSupport().getOrCreate()
##properties
    spark_conf_list = spark_sess.sparkContext.getConf().getAll()
    for key,val in spark_conf_list:
        print key + "=" + val
    #spark_sess.sparkContext.getConf().contains("spark.files")
    #spark_sess.conf.get("spark.files")
    print "Spark WebURL= %s" % spark_sess.sparkContext.uiWebUrl
    ## Spark Files ##
    spark_files_dir = SparkFiles.getRootDirectory()
    print "spark_files_dir= %s" % spark_files_dir
    print "file_in_Spark_dir= %s" % os.listdir(spark_files_dir)
    spark_sess.sql("SET -v").show()
    spark_logger = logging.Log4j(spark_sess)
    return spark_sess,spark_files_dir,spark_logger

class spark_file_handler():
    def read_file(self,spark,format,path,mode="permissive"):
        dfname = spark.read.format(format).load(path)
        #dfname.createOrReplaceTempView("test")
        #print spark.sql('select count(*) as dfcount from test').collect()
        return dfname

    def write_file(self,spark,dfname,format,path,mode="overwrite"):
        dfname.write.format(format).mode(mode).save(path)
        #avdf.select("name", "favorite_color").write.format("com.databricks.spark.avro")

    def read_csv(self,spark,path,options=[]):
        dfname = spark.read.format("csv")\
                        .option("header", "true")\
                        .option("inferSchema", "true")\
                        .load(path)
        #dfname.createOrReplaceTempView("test")
        #print spark.sql('select count(*) as dfcount from test').collect()
        return dfname

    def read_hive(self):
        hive_df = spark.sql("select * from nse")
        return hive_df

    def read_db():
        pass

if __name__ == '__main__':
    #os.unsetenv('PYSPARK_SUBMIT_ARGS')
    #del os.environ["PYSPARK_SUBMIT_ARGS"]
    #print(os.environ["PYSPARK_SUBMIT_ARGS"]) #print(os.environ)["configs/etl_config.json"]

    spark,configpath,log= start_spark("prod","[configs/etl_config.json,/usr/local/Cellar/hive/2.1.0/libexec/conf/hive-site.xml]","test")
#checks and balances
    #rc = python_file_handler().check_zerobyte("rundate=20190914","sec_bhavdata_full.csv")
    #print rc
#reading
    log.info("Read files")
    #avrodf = spark_file_handler().read_file(spark,"com.databricks.spark.avro","resources/users.avro","dropMalformed")
    csvdf = spark_file_handler().read_csv(spark,"hdfs://localhost:9001/lake/files/rundate=20190915/sec_bhavdata_full.csv")
    csvdf1 = spark_file_handler().read_csv(spark,"hdfs://localhost:9001/lake/files/rundate=20191001/sec_bhavdata_full_20191001.csv")
    #csvdf.show(20)
    #parquetdf = spark_file_handler().read_file(spark,"parquet","SparkFinal/resources/namesAndFavColors.parquet","failFast")
    hivedf = spark_file_handler().read_hive()
    hivedf.show()
#cleaning nulls spaces duplicates
    csvdf = csvdf.toDF(*map(str.strip,csvdf.columns)) # trimmed columns names
    csvdf1 = csvdf1.toDF(*map(str.strip,csvdf1.columns)) # trimmed columns names
    trim(csvdf.SYMBOL)
    #map(lambda x: trim('csvdf.' + x) ,csvdf.columns) # trimmed columns values
    #csvdf.count()
    #csvdf.dropna()
    #csvdf.fillna({'SYMBOL':''})
    #csvdf.select(trim(col('SYMBOL'))).show(20)
    csvdf.dropDuplicates()

#generic operations
    #csvdf.schema
    #csvdf.describe()
    #csvdf.distinct()
    #csvdf.explain()
    #csvdf.columns
    #csvdf.printSchema()

#columns options
    #csvdf.SYMBOL
    #csvdf["SYMBOL"]

    #abs(csvdf.SYMBOL)
    #array(csvdf.SYMBOL)
    #csvdf.select(col("SYMBOL")).show(10)
    #csvdf.selectExpr("*").show(10)
    #csvdf.drop("SYMBOL").show(1)
    #csvdf.toDF('SYMBOL').show(1)
    #csvdf.withColumn("NEW_COLUMN",col("SYMBOL")).show(1)
    #csvdf.withColumnRenamed("SYMBOL","NEW_COLUMN_NAME").show(1)
    #csvdf.select(col("SYMBOL").alias("NEW_COLUMN_NAME")).show(1)

# dataframes -> tables -> dataframes ->rdd -> json
    csvdf.createOrReplaceTempView("nse_data")
    #csvdf.toJSON().collect() #rdd
    #csvdf.toPandas()

#filtering

    #csvdf.filter(col("SYMBOL").startswith("KOTAK")).show(10)
    #csvdf.filter(csvdf.SYMBOL.startswith("SYM")).show(10)
    #csvdf.filter(csvdf.SERIES == " EQ").show(10)
    #csvdf.filter(csvdf.SERIES.startswith(" EQ")).show(10)
    #csvdf.where(csvdf["SYMBOL"] == "KOTAKBANK").show()
    #csvdf.where(" SYMBOL = 'KOTAKBANK'").show()
    #csvdf.first()
    #csvdf.take(10)
    #csvdf.limit(10).show()


#sorting
    #csvdf.sort(csvdf.HIGH_PRICE)
    #csvdf.sortWithinPartitions

# transformation
    #csvdf.replace("KOTAKBANK","KOTAK").filter(col("SYMBOL").startswith("KOTAK"))
    #csvdf.select(struct("SYMBOL","SERIES").alias("str_col")).filter(col("SYMBOL").startswith("KOTAK"))
    #csvdf.select(array("SYMBOL","SERIES").alias("arr_col")).filter(col("SYMBOL").startswith("KOTAK")).show(10)

#joins

    #crossJoin
    #intersect
    #subtract
    #csvdf.join(csvdf1,csvdf.SYMBOL == csvdf1.SYMBOL).select(csvdf.SYMBOL,csvdf.DATE1,csvdf.LOW_PRICE,csvdf1.DATE1,csvdf1.LOW_PRICE).show(10)
    #union


#aggregations
    #csvdf.count()
    #count(csvdf.SYMBOL)
    #csvdf.cube()
    #csvdf.rollup()
    #csvdf.groupBy()
    #csvdf.orderBy(csvdf.SYMBOL).show(20)

    #csvdf.select(trim(csvdf.DATE1).alias("DATE1"),trim(csvdf.SERIES).alias("SERIES"),"LOW_PRICE")\
    #     .where("SERIES = 'EQ'")\
#         .rollup("DATE1","SERIES")\
#         .sum("LOW_PRICE")\
#         .orderBy("SERIES").show()
    #csvdf.where("DATE1 = ' 07-Feb-2019'").rollup("DATE1").sum("LOW_PRICE").show()
    #csvdf.where("SERIES = ' EQ'").groupby("DATE1","SERIES").sum("LOW_PRICE").orderBy("SERIES").show()
    #csvdf.agg(avg)
    #csvdf.select(avg(csvdf.OPEN_PRICE)).show(10)
    max(csvdf.OPEN_PRICE)
    mean(csvdf.OPEN_PRICE)
    min(csvdf.OPEN_PRICE)
    sum(csvdf.OPEN_PRICE)


#caching
    #cacheTable
    #clearCache

    #cache()
    #persist
    #unpersist

# parallelism and partitioning
    #coalesce
    #csvdf.repartition(4).show()

#udf

#writing
    #spark_file_handler().write_file(spark,csvdf1,"parquet","hdfs://localhost:9001/lake/files/rundate=20190916")
    #spark.sql("CREATE TABLE newhive as select * from nse_data")
    #csvdf.write.format("parquet").saveAsTable("testhive")
#hive partition

#rdd use
    csvdf.filter(col("SYMBOL").startswith("KOTAK")).rdd.collect()
    csvdf.filter(col("SYMBOL").startswith("KOTAK")).rdd.keyBy(lambda word: word[0]).collect()
    csvdf.filter(col("SYMBOL").startswith("KOTAK")).rdd.mapValues(lambda word: word.upper()).collect()
    csvdf.filter(col("SYMBOL").startswith("KOTAK")).rdd.keys().collect()
    csvdf.filter(col("SYMBOL").startswith("KOTAK")).rdd.values().collect()
    csvdf.filter(col("SYMBOL").startswith("KOTAK")).rdd.collect()
    csvdf.filter(col("SYMBOL").startswith("KOTAK")).rdd.map(lambda x: x).collect()
    csvdf.filter(col("SYMBOL").startswith("KOTAK")).rdd.flatMap(lambda x: x).collect()
    csvdf.filter(col("SYMBOL").startswith("KOTAK")).rdd.map(lambda x: (x, 1)).collect()
    csvdf.filter(col("SYMBOL").startswith("KOTAK")).rdd.map(lambda r: r[0]).flatMap(lambda x: x.split('A')).collect() #flatMap
    csvdf.filter(col("SYMBOL").startswith("KOTAK")).rdd.map(lambda r: r[0]).map(lambda x: x.split('A')).collect() # map()
    csvdf.rdd.flatMap(lambda x: x.split(' '))

#test ui chaining, lineage
    csvdf.select(trim(csvdf.DATE1).alias("DATE1"),trim(csvdf.SERIES).alias("SERIES"),"LOW_PRICE","SYMBOL")\
        .dropna()\
        .repartition(5,"SYMBOL")\
        .join(csvdf1,csvdf.SYMBOL == csvdf1.SYMBOL)\
        .show(10)
    csvdf.select(trim(csvdf.DATE1).alias("DATE1"),trim(csvdf.SERIES).alias("SERIES"),"LOW_PRICE","SYMBOL")\
        .dropna()\
        .sort(csvdf.SYMBOL)\
        .join(csvdf1,csvdf.SYMBOL == csvdf1.SYMBOL)\
        .show(10)
    #input()
