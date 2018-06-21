import pandas as pd
from sklearn import preprocessing
import numpy as np

df = pd.read_csv('raw_data.csv')
close = (df['close'].tolist())[1:]
close.append('')
df['next_close'] = close
min_max_scaler = preprocessing.MinMaxScaler()
df['volume'] = min_max_scaler.fit_transform(np.reshape(df['volume'],(-1,1)))
df = df.drop([len(df['close'])-1])
df.to_csv('stock_data.csv',index=None)
