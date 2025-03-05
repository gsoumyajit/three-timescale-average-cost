import numpy as np
from scipy.special import softmax
from env1 import CustomEnv
from numpy.random import choice,randint
from collections import deque
import sys,os

nS=20
nA=4

logrd="/data4/home/gsoumyajit/avg/data/optimal1/"
if not os.path.exists(logrd):
          os.makedirs(logrd)
run_num = len(next(os.walk(logrd))[2])
fr=open(logrd+str(run_num)+".csv","w")

N=2000000000
env=CustomEnv()
returns=deque(maxlen=100000)
fr.write("timestep\treturn\n")
state=env.reset()
for n in range(N):
    action=env.pol[state]
    next_state,reward,_,_=env.step(action)
    
    returns.append(reward)
    if n%100000==100:
        mean=np.mean(returns)
        fr.write(str(n)+"\t"+str(mean)+"\n")
        fr.flush()
        print(n//100000,":",mean)
    state=next_state
fr.close()




