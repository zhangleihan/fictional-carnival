
import numpy as np
import pickle

with open('/Users/chris/Desktop/Tensor_test/temporal_dynamics_of_cascades_20161015','r') as fid:
  lines = fid.readlines()
'''
#Half and half
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
'''

cas_vel = []
#label = []
cas_cum = []
f_size_cum = []
f_size_vel = []
#threshold = 1000
for l in lines:
  l = np.array([int(w) for w in l.rstrip('\n').split('\t')])
  if len(l) == 63 and l[3:].sum() >= 100:
    cas_vel.append(l[3:])
    f_size_vel.append(l[62])

#label = np.copy(cas_vel)
cas_cum = np.copy(cas_vel)

for i in range(len(cas_cum)):
    cas_cum[i] = np.cumsum(cas_cum[i])
for i in range(len(cas_cum)):
    f_size_cum.append(cas_cum[i][59])

#for i in range(len(label)):
#	for j in range(60):
		#label[i][j] = (1 + np.sign(label[i][j] - threshold))/2

with open('cas_vel.pkl', 'wb') as fid:
  pickle.dump(cas_vel, fid)

with open('cas_cum.pkl', 'wb') as fid:
  pickle.dump(cas_cum, fid)

with open('f_size_cum.pkl', 'wb') as fid:
  pickle.dump(f_size_cum, fid)

with open('f_size_vel.pkl', 'wb') as fid:
  pickle.dump(f_size_vel, fid)
