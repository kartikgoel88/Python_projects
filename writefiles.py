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
