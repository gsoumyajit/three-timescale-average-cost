import numpy as np
import gym
from tqdm import tqdm

if __name__=="__main__":
    nS=10
    nA=3
    R=np.random.randint(1,10,size=(nA,nS,nS))
    Pr=np.random.randint(1,10,size=(nA,nS,nS)).astype(float)
    for i in range(nA):
        for j in range(nS):
            Pr[i,j]=Pr[i,j]/np.sum(Pr[i,j])
    
    iters=1000
    state0=0
    values=np.zeros(nS)
    policy=np.zeros(nS)
    for it in range(iters):
        value=np.zeros(nS)
        for i in range(nS):
            policy[i]=np.argmin([np.dot(Pr[a,i],R[a,i]+values)-values[state0] for a in range(nA)])
            value[i]=np.min([np.dot(Pr[a,i],R[a,i]+values)-values[state0] for a in range(nA)])
        values=np.copy(value)
        print(it,":",values)

    np.save("mdp/R3.npy",R)
    np.save("mdp/Pr3.npy",Pr)
    np.savetxt("mdp/value3",values)
    np.savetxt("mdp/policy3",policy)

class CustomEnv(gym.Env):
    def __init__(self):
        self.R=np.load("mdp/R3.npy")
        self.Pr=np.load("mdp/Pr3.npy")
        self.V=np.loadtxt("mdp/value3")
        self.pol=np.loadtxt("mdp/policy3").astype(int)
        self.nS=10
        self.nA=3
        print(self.V)


    def reset(self):
        self.state=0
        return self.state

    def step(self,action):
        next_state=np.random.choice(self.nS,p=self.Pr[action,self.state])
        reward=self.R[action,self.state,next_state]
        done=False
        self.state=next_state
        return self.state,reward,done,None

    def sample(self,state,action):
        next_state=np.random.choice(self.nS,p=self.Pr[action,state])
        reward=self.R[action,state,next_state]
        done=False
        return next_state,reward,done,None

