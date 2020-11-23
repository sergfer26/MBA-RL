import torch
import numpy as np
from numpy import cos, pi, exp
from numpy.linalg import norm 
from mesa import Agent
from numpy.random import uniform as unif
from .ddpg.ddpg import DDPGagent, hidden_size
from .ddpg.utils import OUNoise
from .ddpg.models import Actor, Critic

delta = 1e-3
dx = np.array([delta, 0.0])
dy = np.array([0.0, delta])
batch_size = 32

class Pedestrian(Agent):
    '''
    A Boid-style flocker agent.

    The agent follows three behaviors to flock:
        - Cohesion: steering towards neighboring agents.
        - Separation: avoiding getting too close to any other agent.
        - Alignment: try to fly in the same direction as the neighbors.

    Boids have a vision that defines the radius in which they look for their
    neighbors to flock with. Their speed (a scalar) and heading (a unit vector)
    define their movement. Separation is their desired minimum distance from
    any other Boid.
    '''
    def __init__(self, unique_id, model, pos, type_):
        '''
        Create a new Boid flocker agent.

        Args:
            unique_id: Unique agent identifyer.
            pos: Starting position
            type_ : sets the agent direction
        '''
        super().__init__(unique_id, model)
        self.pos = pos
        self.dt = 0.1
        self.dir = np.array([(-1) ** type_, 0])
        self.type = type_
        self.vel = np.zeros(2) # actual vel in every step
        self.speed = np.random.normal(1.34, 0.26) # desired speed
        self.scale_t = 0.5
        self.phi = pi/4
        self.c = 1.5
        self.max_speed = 2


    def get_driving_effect(self):
        f = self.speed * self.dir - self.vel
        return f / self.scale_t

    def get_boundary_effect(self):
        U = lambda z: 10.0 * exp( - z/0.2)
        pos = np.array(self.pos)
        x, y = pos
        B = self.model.height
        if y < B/2: 
            B = 0
        b = np.array([x, B])
        u = U(norm(pos - b))
        dUdx = (U(norm(pos - b + dx)) - u)/delta
        dUdy = (U(norm(pos - b + dy)) - u)/delta
        return - np.array([dUdx, dUdy])

    @staticmethod
    def b(r, vel, dt):
        y = (norm(r) + norm(r - vel * dt)) ** 2
        y -= (norm(vel) * dt) ** 2
        # import pdb; pdb.set_trace()
        return np.sqrt(round(y, 2)) /2

    def get_interactive_effect(self):
        V = lambda z : 21.0 * exp(- z/ 0.3)
        neighbors = self.model.space.get_neighbors(self.pos, 1.5, False)
        F = np.zeros(2)
        for ped in neighbors:
            if type(ped) is Goal:
                pass
            else:
                pos1 = np.array(self.pos)
                pos2 = np.array(ped.pos)
                B = self.b(pos1 - pos2, ped.vel, self.dt)
                dVdx = (V(self.b(pos1 - pos2 + dx, ped.vel, self.dt)) - V(B))/delta
                dVdy = (V(self.b(pos1 - pos2 + dy, ped.vel, self.dt)) - V(B))/delta
                f = - np.array([dVdx, dVdy])
                c = self.get_effective_angle(-f)
                F += c * f
        return F
    
    def get_atractive_effect(self):
        pass

    def get_effective_angle(self, f):
        if np.dot(self.dir , f) <= norm(f) * cos(self.phi):
            return 1
        else:
            return self.c

    def get_motion_effect(self):
        f = np.zeros(2)
        f += self.get_driving_effect()
        f += self.get_boundary_effect()
        f += self.get_interactive_effect()
        return f

    def get_new_vel(self):
        w = self.get_motion_effect() * self.dt + self.vel
        if norm(w) <= self.max_speed:
            u = w / norm(w)
            self.vel = self.max_speed * u
        else:
            self.vel = w

    def step(self):
        self.get_new_vel()
        x, y = self.pos + self.vel * self.dt
        new_pos = np.array([x, y])
        self.model.space.place_agent(self, new_pos)


class Robot(DDPGagent):

    def __init__(self, unique_id, model, pos, angle, env, train=True):
        super().__init__(unique_id, model, env)
        # Simulation params
        self.pos = pos
        self.dt = 0.1
        self.speed = 1.34
        self.angle = angle
        self.vel = np.zeros(2)
        # OU process
        self.noise = OUNoise(self.env.action_space)
        self.train = train
        if not self.train:
            self.import_networks()
        # step 
        self.i = 0
        self.a = None

    def import_networks(self):
        num_states = self.env.observation_space.shape[0]
        num_actions = self.env.action_space.shape[0]

        self.actor = Actor(num_states, hidden_size, num_actions)
        self.critic = Critic(num_states + num_actions, hidden_size, num_actions)

        self.actor.load_state_dict(torch.load("crowd_dynamics/ANNS/actor.pth"))
        self.critic.load_state_dict(torch.load("crowd_dynamics/ANNS/critic.pth"))
            
        self.actor.eval()
        self.critic.eval()

    def update_info(self):
        return self.a

    def step(self):
        x, y = self.pos; phi = self.angle
        state = np.array([x, y, phi]) 
        action = self.get_action(state)
        if self.train:
            action = self.noise.get_action(action, self.i)
        new_state, reward, done = self.env.step(self, action)
        if self.train:
            self.memory.push(state, action, reward, new_state, done)
            if len(self.memory) > batch_size: 
                self.update(batch_size)
        self.angle = new_state[2]
        self.model.space.place_agent(self, new_state[0:2])
        self.a = (reward, new_state, done)
        self.i += 1

class Goal(Agent):

    def __init__(self, unique_id, model, pos, type_=None):
        super().__init__(unique_id, model)
        self.pos = pos
        self.type = None

    def step(self):
        pass


        
