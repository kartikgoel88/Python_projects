import pandas as pd
import numpy as np
import os

list = [176.2,158.4,167.6,156.2,161.4]
heights_A = pd.Series([176.2,158.4,167.6,156.2,161.4],index=['s1', 's2', 's3', 's4','s5'])
heights_A
heights_A.shape
heights_A[1]
heights_A.s2
heights_A[1:4]
heights_A[['s2','s3','s4']]

weights_A = pd.Series([85.1, 90.2, 76.8, 80.4,78.9],index=['s1', 's2', 's3', 's4','s5'])
weights_A
weights_A.dtype

df_A = pd.DataFrame({'Student_height':heights_A,'Student_weight':weights_A})
df_A
df_A['Student_height']

heights_B = pd.Series( 25 * np.random.randn(5) + 170,index=['s1', 's2', 's3', 's4','s5'])
weights_B = pd.Series( 12 * np.random.randn(5) + 75,index=['s1', 's2', 's3', 's4','s5'])
heights_B
weights_B
df_B = pd.DataFrame({'Student_height':heights_B,'Student_weight':weights_B})
df_B
 #column
df_B.Student_height
df_B['Student_height']
 #row
df_B.loc[['s2','s1']]
df_B.iloc[1]
#slicing
df_B.index.str.endswith(('1','4'))
df_B.loc[lambda df: df.index.str.endswith(('1','4'))]


#panel
data = {'classA': df_A,'classB': df_B}
p = pd.Panel(data)
p

#write and reads to files
df_A.to_csv('data_fresco_excercise.csv',index=False)
df_A.to_csv('data_fresco_excercise.csv',index=True)
os.path.exists('data_fresco_excercise.csv')
df_A2 = pd.read_csv('data_fresco_excercise.csv')
df_A3 = pd.read_csv('data_fresco_excercise.csv',index_col=0)
df_A4 = pd.read_csv('data_fresco_excercise.csv',header=None)
df_A5 = pd.read_csv('data_fresco_excercise.csv',header=None,skiprows=1)
df_A2
df_A2.index
df_A2.shape
df_A3
df_A3.index
df_A5

#dates
dates = pd.date_range('1-Sep-2017','15-Sep-2017')
dates
datelist = ['14-Sep-2017', '16-Sep-2017']
search_dates = pd.to_datetime(datelist)
search_dates[search_dates.isin(dates)]

#multi index
arraylist = [['classA']*5 + ['classB']*5, ['s1', 's2', 's3','s4', 's5']*2]
mi_index = pd.MultiIndex.from_tuples(arraylist)
arraylist
mi_index.levels

# null values
df_A3.loc['s3'] = np.nan
df_A3.loc['s5','Student_weight'] = np.nan
df_A4 = df_A3.dropna()
df_A4 = df_A3.dropna(subset=['Student_height'])
df_A4.index

#filtering by contents
filter_cond1 = (df_A['Student_height'] > 160.0)  & (df_A['Student_weight']  < 80.0)
df_A[filter_cond1]
df_A.query('Student_height > 160.0')

#filtering by labels
df_A.filter(items=['Student_height'],axis=1)
df_A.filter(regex='5',axis = 0)
df_A[df_A.index.str.endswith('5')]

#column add and grouping
df_A['Gender'] = ['M', 'F', 'M', 'M', 'F']
df_A
df_A.groupby('Gender').mean()
df_A.groupby('Gender').min()
df_A.groupby('Gender').groups
g = df_A.groupby(df_A.index.str.len())
g.filter(lambda x: len(x) > 1)

# adding a row to a dataframe to create a new dataframe
s= pd.Series([165.4, 82.7, 'F'],index= ['Student_height', 'Student_weight', 'Gender'],name='s6')
df_AA = df_A.append(s)
df_AA

# concat 2 df's in a new df
df_B.index = [ 's6', 's7', 's8', 's9', 's10']
df_B['Gender'] = ['F', 'M', 'F', 'F', 'M']
df = pd.concat([df_A,df_B])

# merging and joining 2 dataframes
nameid = pd.Series(range(101, 111))
name = pd.Series(['person' + str(i) for i in range(1, 11)])
master = pd.DataFrame({'nameid':nameid,'name':name})
transaction = pd.DataFrame({'nameid':[108, 108, 108,103], 'product':['iPhone', 'Nokia', 'Micromax', 'Vivo']})
name,nameid
master
transaction
master.merge(transaction,on='nameid')



df = pd.DataFrame({'A':[34, 78, 54], 'B':[12, 67, 43]}, index=['r1', 'r2', 'r3'])
df[:2]
df.iloc[:2]
d = pd.date_range('11-Sep-2017', '17-Sep-2017', freq='2D')
d + pd.Timedelta('1 days 2 hours')
