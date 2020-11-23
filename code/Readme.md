# Autonomous navigation in crowd simulation using DDPG 

An implementation of autonomous navigation using the popular RL algorithm for continuos action spaces known as Deep Deterministic Policy Gradient.  

This model uses the Mesas's continuos space feature, and uses numpy arrays to represent vectors. It also uses popular pytorch framework to develop the networks needed in DDPG.

## Libraries 
* mesa 
* torch 
* tqdm 
* gym
* numpy
* matplotlib
* math 
* collections

## How to Run

Train the model:
```
    $ python training_robot.py
```

Viewing the live training progress (plots): 
```
    $ rm -r runs
    $ tensorboard --logdir=runs
```

Viewing a Mesa simulation: 
```
    $ mesa runserver
```

Then open your browser to [http://127.0.0.1:8888/](http://127.0.0.1:8888/) and press Reset, then Run.

## Files

* crowd_dynamics/agent.py: It models de Pedestrian, Robot and Goal, they are based on Mesa's Agent.
* crowd_dynamics/model.py: It's where the Crowd Dynamics model is coded.
* crowd_dynamics/server.py: It shows the visualization of the enviorment.
* crowd_dynamics/ddpg: You can find the implentation of DDPG based on [Deep Deterministic Policy Gradients Explained](https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b).
* crowd_dynamics/ddpg/crowd_env.py: It shows the implementation of an enviorment based on the Social Force Model using Open Ai's gym.
* crowd_dynamics/ANNS: It is the place where is saved the ddpg networks. The networks are called in Mesa simulation.

