import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

'''fetch data and format it'''
data = fetch_movielens(min_rating=9.0)
print(data)