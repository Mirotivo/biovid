import pandas as pd
import numpy as np
Signals = pd.read_csv('Signals/120716-14_m_20-sad_bio.csv',sep = 	',').iloc[:,:].values
''' Visualize ECG '''
from biosppy.signals import ecg
ts,filtered,rpeaks,templates_ts,templates,heart_rate_ts,heart_rate = ecg.ecg(Signals[:,1],512,True)

ts_mean = np.mean(ts)
ts_std = np.std(ts)
ts_amin = np.amin(ts)
ts_amax = np.amax(ts)
ts_median = np.median(ts)
ts_range = np.max(ts) - np.amin(ts)

filtered_mean = np.mean(filtered)
filtered_std = np.std(filtered)
filtered_amin = np.amin(filtered)
filtered_amax = np.amax(filtered)
filtered_median = np.median(filtered)
filtered_range = np.max(filtered) - np.amin(filtered)

rpeaks_mean = np.mean(rpeaks)
rpeaks_std = np.std(rpeaks)
rpeaks_amin = np.amin(rpeaks)
rpeaks_amax = np.amax(rpeaks)
rpeaks_median = np.median(rpeaks)
rpeaks_range = np.max(rpeaks) - np.amin(rpeaks)

templates_ts_mean = np.mean(templates_ts)
templates_ts_std = np.std(templates_ts)
templates_ts_amin = np.amin(templates_ts)
templates_ts_amax = np.amax(templates_ts)
templates_ts_median = np.median(templates_ts)
templates_ts_range = np.max(templates_ts) - np.amin(templates_ts)

templates_mean = np.mean(templates)
templates_std = np.std(templates)
templates_amin = np.amin(templates)
templates_amax = np.amax(templates)
templates_median = np.median(templates)
templates_range = np.max(templates) - np.amin(templates)

heart_rate_ts_mean = np.mean(heart_rate_ts)
heart_rate_ts_std = np.std(heart_rate_ts)
heart_rate_ts_amin = np.amin(heart_rate_ts)
heart_rate_ts_amax = np.amax(heart_rate_ts)
heart_rate_ts_median = np.median(heart_rate_ts)
heart_rate_ts_range = np.max(heart_rate_ts) - np.amin(heart_rate_ts)

heart_rate_mean = np.mean(heart_rate)
heart_rate_std = np.std(heart_rate)
heart_rate_amin = np.amin(heart_rate)
heart_rate_amax = np.amax(heart_rate)
heart_rate_median = np.median(heart_rate)
heart_rate_range = np.max(heart_rate) - np.amin(heart_rate)

features = [ts_mean,filtered_mean,rpeaks_mean,templates_ts_mean,templates_mean,heart_rate_ts_mean,heart_rate_mean]