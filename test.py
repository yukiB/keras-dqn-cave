from __future__ import division

import argparse
import os
import sys
from multiprocessing import Pool
import multiprocessing as multi
import time
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

from cave import DQNCave
from dqn_agent import DQNAgent

state_num = 5

def init():
    img.set_array(state_t_1)
    plt.axis("off")
    return img,


def gray2rgb(val):
    if val == 0.4:
        return np.array([0.0, 0.8, 1.0])
    if val == 0.5:
        return np.array([0.0, 0.4, 1.0])
    elif val >= 0.8:
        return np.array([1.0, 0.0, 0.0])
    else:
        return np.array([0.0, 0.0, 0.0])

    
def onkey(event):
    global action_t
    if event.key == 'q':
        time.sleep(1)
        plt.close('all')
        sys.exit()

def animate(step):
    global past_time, S
    global state_t_1, reward_t, terminal

    if terminal:
        env.reset()
        S = deque(maxlen=state_num * 2)
        sys.stderr.write('\r\033[K')
        print("SCORE: {0:03d}".format(past_time))
    else:
        state_t = state_t_1
        sys.stderr.write('\r\033[K SCORE: {0:03d}'.format(past_time))

        if len(S) == 0:
            [S.append(state_t) for i in range(state_num * 2)]
        else:
            S.append(state_t)
        tmpS = [S[(s + 1) * 2 - 1] for s in range(state_num)]
        
        

        if reward_t == 1:
            n_catched += 1
        # execute action in environment
        action_t = agent.select_action(tmpS, 0.0)
        env.execute_action(action_t)

    # observe environment
    state_t_1, reward_t, terminal, past_time = env.observe()

    # animate
    shape = state_t_1.shape
    data = state_t_1.ravel()
    p = Pool(multi.cpu_count())
    data = p.map(gray2rgb, data)
    p.close()
    img.set_array(np.array(data).reshape(*shape, 3))    
    plt.axis("off")
    return img,


if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path")
    parser.add_argument("-s", "--save", dest="save", action="store_true")
    parser.set_defaults(save=False)
    args = parser.parse_args()

    print('----------------------')
    print('q: exit')
    print('----------------------')
    # environmet, agent
    env = DQNCave(time_limit=False)
    agent = DQNAgent(env.enable_actions, env.name, env.size, state_num)
    agent.load_model(args.model_path)

    # variables
    n_catched = 0
    state_t_1, reward_t, terminal, past_time = env.observe()
    S = deque(maxlen=state_num * 2)
    # animate
    fig = plt.figure(figsize=(env.screen_n_rows / 10, env.screen_n_cols / 10))
    fig.canvas.set_window_title("{}-{}".format(env.name, agent.name))
    cid = fig.canvas.mpl_connect('key_press_event', onkey)
    img = plt.imshow(state_t_1, interpolation="none", cmap="gray")
    ani = animation.FuncAnimation(fig, animate, init_func=init, interval=(1000 / env.frame_rate), blit=True)

    if args.save:
        # save animation (requires ImageMagick)
        ani_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "tmp", "demo-{}.gif".format(env.name))
        ani.save(ani_path, writer="imagemagick", fps=env.frame_rate)
    else:
        # show animation
        plt.show()
