#!/usr/bin/python
import numpy as np
import pickle

with open('/Users/chris/Desktop/Tensor_test/temporal_dynamics_of_cascades_20161015','r') as fid:
  lines = fid.readlines()
train = []
val = []
for l in lines:
  l = np.array([int(w) for w in l.rstrip('\n').split('\t')])
  if len(l) == 63 and l[3:].sum() >= 100:
    train.append(l[3:33])
    val.append(l[33:])

with open('train_half.pkl', 'wb') as fid:
  pickle.dump(train, fid)

with open('val_half.pkl', 'wb') as fid:
  pickle.dump(val, fid)
