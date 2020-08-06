import pandas as pd
import nsepy
import numpy
from datetime import datetime
import matplotlib.pyplot as plt

df_A = pd.read_csv('../ITC.NS.csv',index_col='Date')
#print(df_A)

df_B = df_A.Close
#df_B.columns =['Close']
#df_B.index = pd.to_datetime(df_B.index)
print(df_B)
#plt.interactive(False)
#df_B.plot()
#plt.show()


df_B['RM'] = df_B.rolling(window=20).mean()
#df_B['rolling_mean2'] = df_B.rolling(window=50).mean()
print(df_B.RM)

plt.plot(df_B.index, df_B, label='AMD')
plt.plot(df_B.index, df_B.rolling_mean, label='AMD 20 Day SMA', color='orange')
plt.plot(df_B.index, df_B.rolling_mean2, label='AMD 50 Day SMA', color='magenta')
plt.legend(loc='upper left')
plt.show()

