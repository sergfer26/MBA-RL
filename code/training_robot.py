import torch 
import numpy as np
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm 
from torch.utils.tensorboard import SummaryWriter
from crowd_dynamics.ddpg.crowd_env import MAXSTEPS
from crowd_dynamics.model import CrowdDynamics


batch_size = 32
EPOCHS = 8
writer_train = SummaryWriter()
writer_test = SummaryWriter()
model = CrowdDynamics(20, 20, 0.3, train=True)
train_time = 0.0


def training_loop(writer, model, steps, pbar, episode, test=False):
    model.robot.noise.reset()
    if test: 
        model.robot.train = False
    else:
        model.robot.train = True
    model.robot.env.reset(model.robot)
    episode_reward = 0.0
    for s in range(steps):
        model.step()
        reward, new_state, done = model.robot.update_info()
        episode_reward += float(reward)
        if s % 5 == 0:
            x, y, phi = new_state
            pbar.set_postfix(reward='{:.3f}'.format(episode_reward), x='{:.3f}'.format(x), y='{:.3f}'.format(y), phi='{:.3f}'.format(phi))
            pbar.update(5)
        if done: 
            writer.add_scalar('episode vs x', new_state[0], episode)
            writer.add_scalar('episode vs y', new_state[1], episode)
            writer.add_scalar('episode vs G_t', episode_reward, episode)
            writer.add_scalar('episode vs r', reward, episode)
            writer.add_scalar('episode vs stop time', s + 1, episode)
            break
    return new_state[0], new_state[1], reward, episode_reward, s+1


X = []; X_ = [] # final x
Y = []; Y_ = [] # final y
R = []; R_ = [] # final reward
ER = []; ER_ = [] # gain by episode
S = []; S_ = [] # stop time


for episode in range(EPOCHS):
    start_time = time()
    with tqdm(total = MAXSTEPS, position=0) as pbar_train:
        pbar_train.set_description(f'Episode {episode + 1}/'+str(EPOCHS)+' - training')
        pbar_train.set_postfix(reward='0.0', x='0.0', y='0.0', phi='0.0')
        x, y, r, er, s = training_loop(writer_train, model, MAXSTEPS, pbar_train, episode)
        train_time +=  time() - start_time
    with tqdm(total = MAXSTEPS, position=0) as pbar_test:
        pbar_test.set_description(f'Episode {episode + 1}/'+str(EPOCHS)+' - test')
        pbar_test.set_postfix(reward='0.0', x='0.0', y='0.0', phi='0.0')
        x_, y_, r_, er_, s_ = training_loop(writer_train, model, MAXSTEPS, pbar_test, episode, test=True)

    X.append(x); X_.append(x_)
    Y.append(y); Y_.append(y_)
    R.append(r); R_.append(r_)
    ER.append(er); ER_.append(er_)
    S.append(s); S_.append(s_)

print("--- %s minutes ---", train_time)

fig = plt.figure(figsize=(10, 10))


plt.subplot(2, 2, 1)
plt.plot(X)
plt.plot(X_)
plt.title('final x position')
plt.ylabel('position (x)')
plt.xlabel('episode')
plt.legend(['train', 'test'], loc='upper left')

plt.subplot(2, 2, 2)
plt.plot(Y)
plt.plot(Y_)
plt.title('final y position')
plt.ylabel('position (y)')
plt.xlabel('episode')
plt.legend(['train', 'test'], loc='upper left')

plt.subplot(2, 2, 3)
plt.plot(R)
plt.plot(R_)
plt.title('final reward')
plt.ylabel('reward')
plt.xlabel('episode')
plt.legend(['train', 'test'], loc='upper left')


plt.subplot(2, 2, 4)
plt.plot(ER)
plt.plot(ER_)
plt.title('episode reward')
plt.ylabel('reward')
plt.xlabel('episode')
plt.legend(['train', 'test'], loc='upper left')

plt.show()


plt.plot(S)
plt.plot(S_)
plt.title('stop time')
plt.ylabel('time')
plt.xlabel('episode')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

torch.save(model.robot.actor.state_dict(), "crowd_dynamics/ANNS/actor.pth")
torch.save(model.robot.critic.state_dict(), "crowd_dynamics/ANNS/critic.pth")





