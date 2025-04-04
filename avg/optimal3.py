import numpy as np
from scipy.special import softmax
from env3 import CustomEnv
from numpy.random import choice,randint
from collections import deque
import sys,os

nS=100
nA=9

logrd="data/optimal3/"
if not os.path.exists(logrd):
          os.makedirs(logrd)
run_num = len(next(os.walk(logrd))[2])
fr=open(logrd+str(run_num)+".csv","w")

N=10000000000
env=CustomEnv()
returns=deque(maxlen=100000)
fr.write("timestep\treturn\n")
state=env.reset()
for n in range(N):
    action=env.pol[state]
    next_state,reward,_,_=env.step(action)
    
    returns.append(reward)
    if n%100000==1000:
        mean=np.mean(returns)
        fr.write(str(n)+"\t"+str(mean)+"\n")
        fr.flush()
        print(n//100000,":",mean)
    state=next_state
fr.close()




