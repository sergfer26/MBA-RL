import gym 
import numpy as np
from scipy.stats import norm as normal
from numpy.linalg import norm
from gym import spaces
from numpy import pi, cos, sin
from math import fmod

MAXSTEPS = 25

class CrowdEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, model, state):
        self.action_space = spaces.Box(low=np.zeros(1), high=2 * pi * np.ones(1))
        self.observation_space = spaces.Box(low=np.array([0, 0, 0]), high=np.array([25, 25, 2 *pi]))
        self.state = state
        self.old_state = np.zeros(2)
        self.dt = 0.1
        self.i = 0
        self.model = model
        self.goal = np.array(self.model.goal.pos)
        # Rewards 
        self.rs = -0.5
        self.rh = -2.5 
        self.rg = 5.0
        self.rp = 2.5
        self.ra = -0.03
        # Thereshold
        self.Ds = 0.5
        self.Dg = 1.0
        self.sigma = 1.0
        self.mu = 0.0

    def r_h(self, z):
        return self.rh * normal(self.mu, self.sigma).pdf(z)
    
    def r_g(self, z, lt):
        if z < self.Dg:
            return self.rg
        else:
            return self.rp * lt

    def reward(self, action):
        R = 0.0 
        neighbors = self.model.space.get_neighbors(self.state[0:2], self.Ds, False)
        d = norm(self.state[0:2] - self.goal) 
        lt = norm(self.old_state[0:2] - self.goal) - d
        R += self.r_g(d, lt)
        R += self.ra * abs(action)
        for ped in neighbors:
            if type(self.model.goal) == type(ped):
                pass 
            else: 
                Y = np.array(ped.pos)
                R += self.r_h(norm(self.state[0:2] - Y))
        
        return R

    def is_done(self):
        if norm(self.state[0:2] - self.goal) < 1:
            return True
        elif self.step == MAXSTEPS - 1:
            return True
        else: 
            return False

    def action2pos(self, agent, action):
        x, y = agent.pos
        phi = agent.angle
        new_phi = self.dt * action + phi
        new_phi = fmod(new_phi, 2 * pi)
        agent.vel = agent.speed * np.array([cos(new_phi), sin(new_phi)])
        s = agent.vel * self.dt
        s += np.array([x, y])
        return s[0], s[1], phi

    def reset(self, agent):
        self.state = self.model.reset()
        agent.pos = self.state[0:2]
        agent.angle = self.state[2]
        self.i = 0
        agent.i = 0 
        self.model.space.place_agent(agent, agent.pos)

    def step(self, agent, action):
        self.old_state = self.state
        x, y, phi = self.action2pos(agent, action)
        self.state = np.array([x, y, phi])
        R = self.reward(action)
        done = self.is_done()
        self.i += 1
        return self.state, R, done


        



            

        

