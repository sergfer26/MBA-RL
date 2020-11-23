from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter
from numpy import sin, cos
from .agent import Robot, Goal
from .model import CrowdDynamics
from .SimpleContinuousModule import SimpleCanvas


def crowd_draw(agent):
    if agent is None:
        return

    portrayal = {'Filled':'true', 'Layer': 0}
    if type(agent) is Goal: 
        portrayal['Shape'] = 'rect'
        portrayal['h'] = 0.02
        portrayal['w'] = 0.02
        portrayal['Color'] = 'Red'
    elif type(agent) is Robot:
        portrayal['Shape'] = "circle"
        portrayal['r'] = 2
        # portrayal['r'] = 2
        # portrayal['text_color'] = 'Black'
        portrayal['Color'] = 'Red'
        # portrayal['stroke_color'] = '#030303'
    else:
        portrayal['Shape'] = 'circle'
        portrayal['r'] = 2
        portrayal['Color'] = 'Blue'
        # portrayal['stroke_color'] = '#030303'
    return portrayal


params = {
    'width': 20, 
    'height': 20, 
    'density': UserSettableParameter('slider', 'crowd density', 0.3, 0.00, 0.9, 0.1),
    'train': False
    }

boid_canvas = SimpleCanvas(crowd_draw, 500, 500)
server = ModularServer(CrowdDynamics, [boid_canvas], "Crowd Dynamics", params)
