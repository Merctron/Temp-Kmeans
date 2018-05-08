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

ginf.info()

# K-Means Configuration
GINF_CONFIG = {
    'diff_cols': [1, 2, 4, 5, 6],
    'diff_labels': [
        'Indoor_Air_Temp_1',
        'Indoor_Air_Temp_2',
        'Outdoor_Air_Temp_1',
        'Supply_Water_Temp_1',
        'Return_Water_Temp_1'
    ]
}

config = kmeans.KmeansConfig(GINF_CONFIG['diff_cols'], 
  GINF_CONFIG['diff_labels'], 1)
kmeansObj = kmeans.Kmeans(config, 10, ginf)
kmeansObj.cluster()