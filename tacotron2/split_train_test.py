import os,sys,json

import random
import numpy as np

np.random.seed(44)

# metadata 
with open("./filelists/vlsp/metadata.txt", "r+") as f:
    metadata = f.read()
    metadata = metadata.strip().split('\n')
    metadata = list(metadata)
    metadata = ["./filelists/vlsp/wavs_train/" + i for i in metadata]

np.random.shuffle(metadata)

N = len(metadata)

N_test  = int(N / 10)
N_train = N- N_test
print(N_train)

train = metadata[:N_train]
test = metadata[N_train:]
val = [i for i in test]

with open("./filelists/vlsp/train.txt","w+") as f:
    f.write("\n".join(train))
with open("./filelists/vlsp/test.txt","w+") as f:
    f.write("\n".join(test))
with open("./filelists/vlsp/val.txt","w+") as f:
    f.write("\n".join(val))