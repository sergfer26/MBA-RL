'''
Crowd Dynamics
=============================================================
A Mesa implementation of autonomous navigation in crowd 
enviorment.
'''

import random
import numpy as np
from numpy import pi
from numpy.random import uniform as unif 
from mesa import Model
from mesa.space import ContinuousSpace
from mesa.time import RandomActivation
from .agent import Pedestrian, Robot, Goal
from .ddpg.crowd_env import CrowdEnv


class CrowdDynamics(Model):
    '''
    Flocker model class. Handles agent creation, placement and scheduling.
    '''

    def __init__(self, width, height, density, train=True):
        '''
        Create a new Flockers model.

        Args:
            N: Number of Boids
            width, height: Size of the space.
            speed: How fast should the Boids move.
            vision: How far around should each Boid look for its neighbors
            separtion: What's the minimum distance each Boid will attempt to
                       keep from any other
        '''
        self.height = height 
        self.width = width
        self.density = density
        self.space = ContinuousSpace(width, height, torus=True)
        self.schedule = RandomActivation(self)
        self.goal = Goal(0, self, (self.width//2, self.height//2))
        self.schedule.add(self.goal)
        self.train = train
        self.agents_setup()
        self.robot_setup()
            

        self.running = True


    def agents_setup(self):
        '''
        Create N agents, with random positions and starting headings.
        '''
        N = int(self.density * self.height * self.width)
        for i in range(1, N + 1):
            x = self.width * self.random.random()
            if self.random.random() < 1/2:
                y = unif(0, self.height / 2)
                direction = False # ->
            else:
                y = unif(self.height / 2, self.height) 
                direction = True # <-
            pos = np.array([x, y])
            agent = Pedestrian(i, self, (x, y), direction)
            self.space.place_agent(agent, pos)
            self.schedule.add(agent)

    def robot_setup(self):
        j = int(self.density * self.height * self.width) + 1
        state = self.reset()
        self.robot = Robot(j, self, state[0:2], state[2], CrowdEnv(self, state), train=self.train)
        self.space.place_agent(self.robot, state[0:2])
        self.schedule.add(self.robot)
    
    def reset(self):
        x = self.width / 2 + unif(-2.5, 2.5)
        y = self.height / 2 + unif(-2.5, 2.5)
        phi = unif(0, 2 * pi)
        return np.array([x, y, phi])

    def step(self):
        self.schedule.step()

