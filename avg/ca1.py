import numpy as np
from scipy.special import softmax
from env1 import CustomEnv
from numpy.random import choice,randint
from collections import deque
import sys,os

nS=20
nA=4

logrd="/data4/home/gsoumyajit/avg/data/ca1/"
if not os.path.exists(logrd):
          os.makedirs(logrd)
run_num = len(next(os.walk(logrd))[2])
fr=open(logrd+str(run_num)+".csv","w")

a=lambda n:1*np.log(n+2)/(n+2)
b=lambda n:1/(n+1)
c=lambda n:1/(np.log(n+2)*(n+2))
K=20
value=np.zeros(nS)
theta=np.zeros((nS,nA))
lam=0

N=2000000000
env=CustomEnv()
state0=0
returns=deque(maxlen=100000)
vstep=np.ones(nS)
pstep=np.ones((nS,nA))
fr.write("timestep\treturn\tverror\tlerror\n")
state=env.reset()
for n in range(N):
    probs=softmax(theta[state])
    action=choice(nA,p=probs/np.sum(probs))
    next_state,reward,_,_=env.step(action)
    
    state1=randint(nS)
    probs=softmax(theta[state1])
    action1=choice(nA,p=probs/np.sum(probs))
    next_state1,reward1,_,_=env.sample(state1,action1)

    state2=randint(nS)
    action2=randint(nA)
    next_state2,reward2,_,_=env.sample(state2,action2)

    value[state1]+=b(vstep[state1]//K+1)*(reward1+value[next_state1]*(next_state1!=state0)-value[state1]-lam)
    theta[state2,action2]-=a(pstep[state2,action2]//K+1)*(reward2+value[next_state2]*(next_state2!=state0)-value[state2]-lam)
    lam+=c(n//K+1)*value[state0]
    vstep[state1]+=1
    pstep[state2,action2]+=1
    returns.append(reward)
    if n%100000==100:
        mean=np.mean(returns)
        error2=np.linalg.norm(env.V-value-env.V[state0])
        error3=abs(env.V[state0]-lam)
        fr.write(str(n)+"\t"+str(mean)+"\t"+str(error2)+"\t"+str(error3)+"\n")
        fr.flush()
        print(n,":",mean,error2,error3)
    state=next_state
fr.close()




