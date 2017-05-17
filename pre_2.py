#!/usr/bin/python
import numpy as np
import pickle
with open('/Users/chris/Desktop/Tensor_test/data/cas_cum.pkl','rb') as f1:
	cas_cum = pickle.load(f1)

with open('/Users/chris/Desktop/Tensor_test/data/f_size_cum.pkl','rb') as f2:
	f_size = pickle.load(f2)

train = []
val = []

start = 0
step = 7
k = 18214
threshold = 1193
sorted_indices = np.argsort(f_size)
topk_cas = cas_cum[sorted_indices[:-k-1:-1],:]
lesk_cas = cas_cum[sorted_indices[start:start+k*step:step],:]

for i in range(k):
    train.append(lesk_cas[i])
    train.append(topk_cas[i])

for i in range(len(train)):
    flag = (1 + np.sign(train[i][59] - threshold + 1))/2
    val.append([flag,1-flag])


l = int(len(train)*0.8)

train_x = []
val_x = []
train_y = []
val_y = []


for i in range(l):
	train_x.append(train[i][:10])
	train_y.append(val[i])
for i in range(l,len(train)):
	val_x.append(train[i][:10])
	val_y.append(val[i])

with open('train_x.pkl', 'wb') as fid:
	pickle.dump(train_x, fid)

with open('val_x.pkl', 'wb') as fid:
	pickle.dump(val_x, fid)

with open('train_y.pkl', 'wb') as fid:
  pickle.dump(train_y, fid)

with open('val_y.pkl', 'wb') as fid:
  pickle.dump(val_y, fid)
