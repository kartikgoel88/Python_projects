
import os
import csv
import pandas as pd
import numpy as np
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.functions import concat,concat_ws,explode,split,expr, col, column,lit,pow,round,bround,corr,count,mean, stddev_pop, min, max,monotonically_increasing_id
from pyspark.sql.types import StructField, StructType, StringType, LongType,IntegerType,DataType
import pyspark.sql.functions as fspark
import collections

spark = SparkSession.builder \
                    .master("local")\
                    .appName('abc') \
                    .config("spark.some.config.option", "some-value")\
                    .enableHiveSupport()\
                    .getOrCreate()
spark.conf
spark.range(3).show()

myManualSchema = StructType([
  StructField("some", StringType(), True),
  StructField("col", StringType(), True),
  StructField("names", LongType(), False)
])
myRow = Row("Hello", "hi", 1)
myDf = spark.createDataFrame([myRow], myManualSchema)
myDf.show()
myDf.select("col")
myDf.select(expr("col as c"),col("col").alias("c1"),lit(1)).show()
myDf.select(expr("col as c"),col("col").alias("c1"),lit(1)).distinct().count()
myDf.selectExpr("*","col as newColumnName", "col as wer" ).show(2)
myDf.selectExpr("col as `This Long Column-Name`" ).show(2)
myDf.withColumn("numberOne", lit(1) * expr("names")).show(2)
myDf.withColumn("This Long Column-Name",expr("col")).show()
myDf.withColumnRenamed("col", "dest").columns
myDf.drop("col").columns
myDf.filter(col("names") == 1).show()
myDf.filter("names == 1").show()
myDf.where("names == 1").where("some == 'Hello'").show()
myDf.sample(False,.5,5).show()
myDf1 = myDf.randomSplit([0.25, 0.75], 5)
myDf1[0].show()
myDf1[1].show()

#os.listdir(u'')
#spark definitive guide chapter 6
df = spark.read.format("csv")\
  .option("header", "true")\
  .option("inferSchema", "true")\
  .load("/Users/kkartikgoel/dev/Python_Projects/2010-12-01.csv")
df.printSchema()

df.show(3)
#df.select(lit(5), lit("five"), lit(5.0))
df.createOrReplaceTempView("dfTable")

df.where(col("InvoiceNo") == 536365).select("InvoiceNo", "Description").show(5, False)
df.where("InvoiceNo <> 536365").show(5, False)
df.withColumn("isExpensive", expr("NOT UnitPrice <= 250")).where("isExpensive").select("Description", "UnitPrice").show(5)
df.where(col("Description").eqNullSafe("hello")).show()

fabricatedQuantity = pow(col("Quantity") * col("UnitPrice"), 2) + 5
df.select(expr("CustomerId"), fabricatedQuantity.alias("realQuantity")).show(2)
df.selectExpr("CustomerId","(POWER((Quantity * UnitPrice), 2.0) + 5) as realQuantity").show(2)

# round
df.select(round(lit(2.5))).show(2)

#stats
df.stat.corr("Quantity", "UnitPrice")
df.select(corr("Quantity", "UnitPrice")).show()
df.describe().show()
df.stat.crosstab("StockCode", "Quantity").show()
df.stat.freqItems(["StockCode", "Quantity"]).show()
df.select(monotonically_increasing_id(),"*").show(2)
#strings
#dates
#complex type  structs, arrays, and maps.

df.select(split(col("Description"), " ").alias("array_col")).selectExpr("array_col[0]").show(2)
df.withColumn("joinedcol", concat(col("Description"),col("InvoiceNo"))).select("joinedcol").show(2,False)
df.withColumn("joinedcol", concat_ws('-',col("Description"),col("InvoiceNo"))).select("joinedcol").show(2,False)

df.selectExpr("struct(Description, InvoiceNo) as complex", "*").show(3,False)
df.withColumn("splitted", split(col("Description"), " "))\
  .withColumn("exploded", explode(col("splitted")))\
  .select("Description", "InvoiceNo", "exploded","splitted").show(2,False)


from pyspark.sql.functions import create_map,udf
df.select(create_map(col("Description"), col("InvoiceNo")).alias("complex_map")).show()
df.select(create_map(col("Description"), col("InvoiceNo")).alias("complex_map")).selectExpr("complex_map['WHITE METAL LANTERN']").show(2)
df.select(create_map(col("Description"), col("InvoiceNo")).alias("complex_map")).select(expr("explode(complex_map)")).show(2)
e = df.Description

def cube(num):
    return num * 3
df.show()
udf_f=udf(cube)
df.select(df.InvoiceNo.cast(IntegerType()))

df.select(udf_f(df.InvoiceNo.cast(IntegerType()))).show(2)
e1 = create_map([df.Description,df.InvoiceNo]).alias("test")
e2 = create_map(col("Description"),col("InvoiceNo")).alias("test1")
e1
e2
df.select(e).show(2)

#JSON

df.toJSON().first()
df.toPandas()

#UDFS

#aggregations

df.groupBy("InvoiceNo", "CustomerId").count().show()
df.groupBy("InvoiceNo").agg(count("Quantity").alias("quan"),expr("count(Quantity)")).show()
df.groupBy("InvoiceNo").agg(expr("avg(Quantity)"),expr("stddev_pop(Quantity)")).show()

#windows

from pyspark.sql.functions import col, to_date
dfWithDate = df.withColumn("date", to_date(col("InvoiceDate")))
dfWithDate.createOrReplaceTempView("dfWithDate")
from pyspark.sql.window import Window
from pyspark.sql.functions import desc
from pyspark.sql.functions import max
from pyspark.sql.functions import sum as _sum
from pyspark.sql.functions import dense_rank, rank

windowSpec = Window\
  .partitionBy("CustomerId", "date")\
  .orderBy(desc("Quantity"))\
  .rowsBetween(Window.unboundedPreceding, Window.currentRow)
windowSpec
maxPurchaseQuantity = max(col("Quantity")).over(windowSpec)

purchaseDenseRank = dense_rank().over(windowSpec)
purchaseRank = rank().over(windowSpec)
dfWithDate.where("CustomerId IS NOT NULL").orderBy("CustomerId")\
  .select(
    col("CustomerId"),
    col("date"),
    col("Quantity"),
    purchaseRank.alias("quantityRank"),
    purchaseDenseRank.alias("quantityDenseRank"),
    maxPurchaseQuantity.alias("maxPurchaseQuantity")).show(30)

#rollup
dfNoNull = dfWithDate.drop()
dfNoNull.createOrReplaceTempView("dfNoNull")
dfNoNull.show()
rolledUpDF = dfNoNull.rollup("Date", "Country").agg(_sum("Quantity"))\
   .selectExpr("Date", "Country", "`sum(Quantity)` as total_quantity").orderBy("Date")
rolledUpDF.show()
rolledUpDF.where("Country IS NULL").show()
rolledUpDF.where("Date IS NULL").show()

#cubes
dfNoNull.cube("Date", "Country").agg(_sum(col("Quantity")))\
  .select("Date", "Country", "sum(Quantity)").orderBy("Date").show()
###################



## rdd and dataframes
a = spark.range(10).rdd
b = spark.range(10).toDF("id").rdd.map(lambda row: row[0])
a.take(4)

sqldf = SparkContext.createDataFrame(ratings)

## using spark dataframes
df = spark.read.csv('KCLT.csv',header=True)
df1 = spark.read.load('KCLT.csv',format='com.databricks.spark.csv', header='true',inferSchema='true')
type(df)
df.describe().show()
df.dtypes
df.count()
#change data type and column name
df = df.withColumn('date',df.date.cast('timestamp'))

## using spark rdd
sc = SparkContext('local','example')
rdd = sc.textFile('KCLT.csv').map(lambda line: line.split(",")[1])
ratings = sc.textFile('../Python_Projects/ml-100k/u.data').map(lambda line: line.split()[2])
rdd.take(4)
ratings.take(5)
rows.take(4)
result = ratings.countByValue()
result = ratings.count()
result = ratings.countBykey()
resultmap = ratings.map(lambda a: len(a))
resultreduce = resultmap.reduce(lambda a,b : a + b)
resultgroup = ratings.groupBy(lambda a: a)
resultgroup.take(5)
result

sortedRes = collections.OrderedDict(sorted(result.items()))
for k,v in sortedRes.iteritems():
    print("%s %i" %(k,v))


#----------
from random import random,seed
from pyspark.sql import SparkSession
from __future__ import print_function
from pyspark.sql.readwriter import DataFrameWriter

spark = SparkSession \
     .builder \
     .appName("Python Spark SQL basic example") \
     .config("spark.some.config.option", "some-value") \
     .getOrCreate()
r1 = rand(seed=10)

df.write.save("result.txt")

# Write chunks of text data
with open('somefile.txt', 'wt') as f:
    f.write(text1)
    f.write(text2)
    ...

# Redirected print statement
with open('somefile.txt', 'wt') as f:
    print(line1, file=f)
    print(line2, file=f)
    ...


#----------Readfiles
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
