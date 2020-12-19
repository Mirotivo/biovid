import heartbeat as hb

data = hb.get_data('data.csv') 
fs = 100.0 #example file 'data.csv' is sampled at 100.0 Hz
print('amr')
measures = hb.process(data, fs)


#Alternatively, use dictionary stored in module
print(hb.measures['bpm']) # returns LF:HF ratio
print(hb.measures['lf/hf']) # returns LF:HF ratio

#You can also use Pandas if you so desire
import pandas as pd
df = pd.read_csv("data.csv")
measures = hb.process(df['hr'].values, fs)
print(measures['bpm'])
print(measures['lf/hf'])