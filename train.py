import numpy as np

import argparse
from cave import DQNCave
from dqn_agent import DQNAgent
import copy
from collections import deque


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path")
    parser.add_argument("-l", "--load", dest="load", action="store_true", default=False)
    parser.add_argument("-e", "--epoch-num", dest="n_epochs", default=200, type=int)
    parser.add_argument("-g", "--graves", dest="graves", action="store_true", default=False, help='Use RmpropGraves (default: off)')
    parser.add_argument("-d", "--ddqn", dest="ddqn", action="store_true", default=False, help='Use Double DQN (default: off)')
    args = parser.parse_args()

    # parameters
    n_epochs = args.n_epochs
    state_num = 4

    # environment, agent
    env = DQNCave()
    agent = DQNAgent(env.enable_actions, env.name, env.size, state_num, graves=args.graves, ddqn=args.ddqn)
    agent.init_model()
    if args.load:
        agent.load_model(args.model_path)

    # variables

    e = 0
    total_frame = 0
    while e < n_epochs:
        # reset
        frame = 0
        loss = 0.0
        do_replay_count = 0
        Q_max = 0.0
        env.reset()
        state_t_1, reward_t, terminal, past_time = env.observe()
        S = deque(maxlen=state_num * 2)

        while not terminal:
            state_t = state_t_1

            if len(S) == 0:
                [S.append(state_t) for i in range(state_num * 2)]
            else:
                S.append(state_t)
                # execute action in environment
            tmpS = [S[(s + 1) * 2 - 1] for s in range(state_num)]
            if frame % 3 == 0:
                action_t = agent.select_action(S, agent.exploration)
                #action_t = agent.select_action([state_t], agent.exploration)
            env.execute_action(action_t)

            # observe environment
            state_t_1, reward_t, terminal, past_time = env.observe()

            # store experience
            start_replay = False
            if frame % 4 == 0 or reward_t == -1:
                new_S = copy.copy(S)
                new_S.append(state_t_1)
                tmpnew_S = [new_S[(s + 1) * 2 - 1] for s in range(state_num)]
                start_replay = agent.store_experience(tmpS, action_t, reward_t, tmpnew_S, terminal)
                #start_replay = agent.store_experience([state_t], action_t, reward_t, [state_t_1], terminal)

            # experience replay
            if start_replay:
                do_replay_count += 1
                if do_replay_count % 3 == 0 or reward_t == -1:
                    agent.update_exploration(e)
                    agent.experience_replay()
                    do_replay_count = 0

            # update target network
            if total_frame % 5000 == 0 and start_replay:
                agent.update_target_model()

            # for log
            frame += 1
            total_frame += 1
            loss += agent.current_loss
            Q_max += np.max(agent.Q_values(tmpS))

        if start_replay:
            print("EPOCH: {:03d}/{:03d} | SCORE: {:03d} | LOSS: {:.4f} | Q_MAX: {:.4f}".format(
                e, n_epochs - 1, past_time, loss / frame, Q_max / frame))
        if e > 0 and e % 5000 == 0:
            agent.save_model(e)
            agent.save_model()
        if start_replay:
            e += 1

    # save model
    agent.save_model()
