import pandas as pd

df = pd.read_csv('raw_data.csv')
close = (df['close'].tolist())[1:]
df = df.drop([len(df['close'])-1])
df.to_csv('stock_data.csv',index=None)
