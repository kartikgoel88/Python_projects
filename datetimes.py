from datetime import datetime ,timedelta,tzinfo
from dateutil.parser import parse
import pandas as pd
#hello!
#python datetime objects

import sys
print(sys.version)
datetime.now()
datetime.now().year
datetime.now().month
datetime.now().day
birthday = datetime(1988,3,8)
#difference and timedelta object
diff = datetime.now() - birthday
diff
diff.days
diff.seconds
diff.days / 365
birthday + timedelta(365)
datetime.now().tzinfo
#datetime to string
str(datetime.now())
datetime.now().strftime('%Y-%m-%d %H:%M:%S %w %z %F %D %a %A')
#python string to datetime
datetime.strptime("1988-03-08","%Y-%m-%d")
parse("1988-03-08") # pandas
parse("1988-03-08", dayfirst=True) # pandas
pd.to_datetime("1988-03-08")
pd.to_datetime(["1988-03-08"])
pd.to_datetime(["1988-03-08"] + [None]) #NaT
pd.to_timedelta(['1 hours']) + birthday

#pandas series and timeseries index - selection, duplicate indices


#Date Ranges, Frequencies, and Shifting - pandas
pd.date_range("1988-03-08","2018-03-08",freq='365d')
pd.date_range("1988-03-08",periods = 10,freq='365d',normalize=True)
