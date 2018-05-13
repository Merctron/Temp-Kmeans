from __future__ import division
import numpy as np
import pandas as pd
import random
import sys
import math
import kmeans

ginf = pd.read_csv('./tagged_data_01_2017.csv', delimiter=';')

cols = list(ginf.columns)
ginf[cols[1:]] = ginf[cols[1:]].fillna(0).astype('float32')

# ginf.info()
ginf.set_index('time', inplace = True)
ginf.index = pd.to_datetime(ginf.index, format='%d/%m/%y %H:%M')
data_ind = ginf.filter(regex="^.*indoor...*$")
data_ind_means=data_ind.groupby(data_ind.index.day).agg(lambda x:np.nanmean(x[x<100]))

data_ind_means_numpy = data_ind_means.as_matrix()
data_ind_means_num = np.reshape(data_ind_means_numpy, -1)

df = pd.DataFrame(data_ind_means_num)
df.info()

#K-Means Configuration
GINF_CONFIG = {
    'diff_cols': [0],
    'diff_labels': [
        'Indoor_Air_Temp'
    ]
}

config = kmeans.KmeansConfig(GINF_CONFIG['diff_cols'], GINF_CONFIG['diff_labels'], 0)
kmeansObj = kmeans.Kmeans(config, 4, ginf)
kmeansObj.cluster()