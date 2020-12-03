import numpy as np
import pandas as pd

print('Loading data ...')

train = pd.read_csv('datas/train_2016.csv')
prop = pd.read_csv('datas/properties_2016.csv')
sample = pd.read_csv('datas/sample_submission.csv')

print('Binding to float32')

for c, dtype in zip(prop.columns, prop.dtypes):
	if dtype == np.float64:
		prop[c] = prop[c].astype(np.float32)

sample['parcelid'] = sample['ParcelId']
df_test = sample.merge(prop, on='parcelid', how='left')
