#!/usr/bin/python
import numpy as np
import pickle

with open('/Users/chris/Desktop/Tensor_test/data/train_half.pkl','rb') as f1:
	train = pickle.load(f1)

with open('/Users/chris/Desktop/Tensor_test/data/val_half.pkl','rb') as f2:
	val = pickle.load(f2)

'''
with open('/Users/chris/Desktop/Tensor_test/data/train.pkl','rb') as f1:
	train = pickle.load(f1)

with open('/Users/chris/Desktop/Tensor_test/data/val.pkl','rb') as f2:
	val = pickle.load(f2)
'''
train_x = []
val_x = []
train_y = []
val_y = []

l = int(len(train)*0.8)



for i in range(l):
	train_x.append(train[i])
	train_y.append(val[i])
for i in range(l,len(train)):
	val_x.append(train[i])
	val_y.append(val[i])

with open('train_x.pkl', 'wb') as fid:
	pickle.dump(train_x, fid)

with open('val_x.pkl', 'wb') as fid:
	pickle.dump(val_x, fid)

with open('train_y.pkl', 'wb') as fid:
  pickle.dump(train_y, fid)

with open('val_y.pkl', 'wb') as fid:
  pickle.dump(val_y, fid)
