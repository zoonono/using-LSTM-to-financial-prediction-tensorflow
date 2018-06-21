###using-LSTm-to-financial-prediction

主题：Long Short Term Memory (LSTM)长短期记忆神经网络去预测金融市场价格

细节：使用过去20天的历史数据（开盘价，收盘价，最高价，最低价，交易量）对后一天的收盘价进行回归预测

1、raw_data.csv 为通过Tushare中获取的股票数据

2、preprocess_data.py对原始数据集中的交易量进行归一化，生成stock_data.csv

3、lstm.py搭建LSTM网络并进行预测，将注释删除可以查看每次训练20次后的预测价格（可以根据自己思路修改相应参数）
