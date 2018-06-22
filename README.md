## using-LSTm-to-financial-prediction<br>
##### 主题：使用Long Short Term Memory (LSTM)长短期记忆神经网络去预测金融市场股票价格<br>
##### 细节：过去20天的历史数据（开盘价，收盘价，最高价，最低价，交易量）进行回归预测<br>
##### 1、raw_data.csv 为通过Tushare中获取的股票数据<br>
##### 2、preprocess_data.py对原始数据集中的交易量进行归一化，生成stock_data.csv<br>
##### 3、lstm.py搭建LSTM网络，将注释删除可以查看预测价格<br>
