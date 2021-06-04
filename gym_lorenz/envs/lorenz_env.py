import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

class LorenzEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.r = 28.
        self.sigma = 10.0
        self.b= 8.0 / 3.0

        self.t = 0
        self.dt = .05

        #action space: a=-1,0,1
        #self.action_space = spaces.Discrete(3)
        self.action_space = spaces.Box(
            low=-5,
            high=5, 
            shape=(1,),dtype=np.float32
        )

        #observation space: x,y, and z, ubounded
        high = np.array([ np.finfo(np.float32).max,
                          np.finfo(np.float32).max,
                          np.finfo(np.float32).max,
                        ],dtype=np.float32)
        self.observation_space = spaces.Box(-high, high,dtype=np.float32)

        #self.seed()

    #rhs of lorenz system
    def f(self,x,a=0):
        return np.stack([
            self.sigma * (x[1] - x[0]),
            x[0] * (self.r - x[2]) - x[1] + a[0],
            x[0] * x[1] - self.b * x[2]
            ],axis=-1)

    #RK-45 integrator
    def integrate(self,x,a=0):
        k1 = self.dt*self.f(x,a)
        k2 = self.dt*self.f(x + 0.5*k1,a)
        k3 = self.dt*self.f(x + 0.5*k2,a)
        k4 = self.dt*self.f(x + k3,a)

        return x+(1.0/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    def seed(self, seed=None):
            self.np_random, seed = seeding.np_random(seed)
            return [seed]
    
    #do one RK step using the action a, return new state and reward
    def step(self, a):
        past = self.x.copy()
        self.x = self.integrate(self.x,a-1)
        self.t += self.dt

        reward = -float((past[0]*self.x[0])<0) #-abs(a[0])*2e-3
        return self.x, reward, False, {}

    #initialize to random
    def reset(self):
        self.t=0
        self.x = 20*np.random.rand(3)
        self.x[0]-=10
        self.x[1]-=10
        return self.x

    def render(self, mode='human'):
        pass 
    def close(self):
        pass
