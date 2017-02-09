import numpy as np

import argparse
from cave import DQNCave
from dqn_agent import DQNAgent
from collections import deque
import copy


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path")
    parser.add_argument("-l", "--load", dest="load", action="store_true", default=False)
    parser.add_argument("-e", "--epoch-num", dest="n_epochs", default=200, type=int)
    args = parser.parse_args()

    # parameters
    n_epochs = args.n_epochs
    state_num = 4

    # environment, agent
    env = DQNCave()
    agent = DQNAgent(env.enable_actions, env.name, env.size)
    if args.load:
        agent.load_model(args.model_path)
    else:
        agent.init_model()
    # variables
    win = 0

    for e in range(n_epochs):
        # reset
        frame = 0
        loss = 0.0
        Q_max = 0.0
        env.reset()
        state_t_1, reward_t, terminal, past_time = env.observe()
        S = deque(maxlen=state_num)

        while not terminal:
            state_t = state_t_1

            if len(S) == 0:
                [S.append(state_t) for i in range(state_num)]
            else:
                S.append(state_t)
                # execute action in environment
            if frame % 3 == 0:
                action_t = agent.select_action(S, agent.exploration)
            env.execute_action(action_t)

            # observe environment
            state_t_1, reward_t, terminal, past_time = env.observe()

            # store experience
            if frame % 3 == 0:
                new_S = copy.copy(S)
                new_S.append(state_t_1)
                agent.store_experience(S, action_t, reward_t, new_S, terminal)

            # experience replay
            agent.experience_replay()

            # for log
            frame += 1
            loss += agent.current_loss
            Q_max += np.max(agent.Q_values(S))

        print("EPOCH: {:03d}/{:03d} | SCORE: {:03d} | LOSS: {:.4f} | Q_MAX: {:.4f}".format(
            e, n_epochs - 1, past_time, loss / frame, Q_max / frame))
        if e > 0 and e % 100 == 0:
            agent.save_model(e)

    # save model
    agent.save_model()
