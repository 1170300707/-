from pandas import read_csv
from datetime import datetime
import pandas as pd
def parse(x):
    return datetime.strptime(x,"%Y-%m-%d|%H:%M:%S.%f")

dataset = read_csv('data/DATA.csv', parse_dates = ['time'], index_col ='time', date_parser = parse)


low = pd.Timestamp(datetime(2017,1,20,hour = 0,minute = 0,second = 0))
high = pd.Timestamp(datetime(2017,1,23,hour = 0,minute = 0,second = 0))


dataset.columns = ['A', 'B', 'C', 'D']
dataset.index.name = 'data'


i = 0
for da in dataset.index:
    if(da <= low):
        dataset.drop([str(dataset.index[i])])
    if(da >= high):
        dataset.drop(str(dataset.index[i]))
    i += 1
dataset.to_csv('data/expData.csv')

