from collections import deque
import os
import random
import copy
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import InputLayer, Convolution2D
from keras.models import model_from_yaml
from keras.optimizers import RMSprop
try:
    from keras.optimizers import RMSpropGraves
except:
    print('You do not have RMSpropGraves')
   
import keras.callbacks
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
from util import clone_model

f_log = './log'
f_model = './models'

model_filename = 'dqn_model.yaml'
weights_filename = 'dqn_model_weights.hdf5'

INITIAL_EXPLORATION = 1.0
FINAL_EXPLORATION = 0.1
EXPLORATION_STEPS = 1000000


def loss_func(y_true, y_pred):
    error = tf.abs(y_pred - y_true)
    quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
    linear_part = error - quadratic_part
    loss = tf.reduce_sum(0.5 * tf.square(quadratic_part) + linear_part)
    return loss
        

class DQNAgent:
    """
    Multi Layer Perceptron with Experience Replay
    """

    def __init__(self, enable_actions, environment_name, env_size, state_num, graves=False, ddqn=False):
        # parameters
        self.name = os.path.splitext(os.path.basename(__file__))[0]
        self.environment_name = environment_name
        self.enable_actions = enable_actions
        self.n_actions = len(self.enable_actions)
        self.minibatch_size = 32
        self.env_size = env_size
        self.replay_memory_size = 10000
        self.learning_rate = 0.00025
        self.discount_factor = 0.9
        self.use_graves = graves
        self.use_ddqn = ddqn
        self.state_num = state_num
        self.exploration = INITIAL_EXPLORATION
        self.exploration_step = (INITIAL_EXPLORATION - FINAL_EXPLORATION) / EXPLORATION_STEPS
        self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        self.model_name = "{}.ckpt".format(self.environment_name)

        self.old_session = KTF.get_session()
        self.session = tf.Session('')
        KTF.set_session(self.session)

        # replay memory
        self.D = deque(maxlen=self.replay_memory_size)
        self.MaxD = []
        self.high_score = 0
        
        # variables
        self.current_loss = 0.0

    def init_model(self):

        self.model = Sequential()
        self.model.add(InputLayer(input_shape=(self.state_num, *self.env_size)))
        self.model.add(Convolution2D(16, 4, 4, border_mode='same', activation='relu', subsample=(2, 2)))
        self.model.add(Convolution2D(32, 2, 2, border_mode='same', activation='relu', subsample=(1, 1)))
        self.model.add(Convolution2D(32, 2, 2, border_mode='same', activation='relu', subsample=(1, 1)))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(self.n_actions, activation='linear'))
        
        
        optimizer = RMSprop if not self.use_graves else RMSpropGraves
        self.model.compile(loss=loss_func,
                           optimizer=optimizer(lr=self.learning_rate),
                           metrics=['accuracy'])

        self.target_model = copy.copy(self.model)


    def update_exploration(self, num):
        if self.exploration > FINAL_EXPLORATION:
            self.exploration -= self.exploration_step * num
            if self.exploration < FINAL_EXPLORATION:
                self.exploration = FINAL_EXPLORATION        

    def Q_values(self, states, isTarget=False):
        # Q(state, action) of all actions
        model = self.target_model if isTarget else self.model
        res = model.predict(np.array([states]))

        return res[0]

    def update_target_model(self):
        self.target_model = clone_model(self.model)

    def select_action(self, states, epsilon):
        if np.random.rand() <= epsilon:
            # random
            return np.random.choice(self.enable_actions)
        else:
            # max_action Q(state, action)
            return self.enable_actions[np.argmax(self.Q_values(states))]

    def store_experience(self, states, action, reward, states_1, terminal, score=None):
        self.D.append((states, action, reward, states_1, terminal))
        start_replay = (len(self.D) >= self.replay_memory_size)
        if start_replay and score and reward == -1\
           and score > self.high_score:
            self.high_score = score
            self.maxD = [self.D[i] for i in range(len(self.D)-150, len(self.D)-10)] if len(self.D) > 150 else copy.copy(self.D)
            self.experience_replay_core(self.maxD, False)
        return start_replay

    def experience_replay(self):
        self.experience_replay_core(self.D)

    def experience_replay_core(self, D, random=True):
        state_minibatch = []
        y_minibatch = []
        action_minibatch = []

        # sample random minibatch
        if random:
            minibatch_size = min(len(D), self.minibatch_size)
            minibatch_indexes = np.random.randint(0, len(D), minibatch_size)
        else:
            minibatch_indexes = range(len(D))

        for j in minibatch_indexes:
            state_j, action_j, reward_j, state_j_1, terminal = D[j]
            action_j_index = self.enable_actions.index(action_j)

            y_j = self.Q_values(state_j)

            if terminal:
                y_j[action_j_index] = reward_j
            else:
                # reward_j + gamma * max_action' Q(state', action') alpha(learing rate) = 1
                if not self.use_ddqn:
                    v = np.max(self.Q_values(state_j_1, isTarget=True))
                else:
                    v = self.Q_values(state_j_1, isTarget=True)[action_j_index]
                y_j[action_j_index] = reward_j + self.discount_factor * v   # NOQA

            state_minibatch.append(state_j)
            y_minibatch.append(y_j)
            action_minibatch.append(action_j_index)

        # training
        self.model.fit(np.array(state_minibatch), np.array(y_minibatch), verbose=0)

        # for log
        score = self.model.evaluate(np.array(state_minibatch), np.array(y_minibatch), verbose=0)
        self.current_loss = score[0]

    def load_model(self, model_path=None):

        yaml_string = open(os.path.join(f_model, model_filename)).read()
        self.model = model_from_yaml(yaml_string)
        self.model.load_weights(os.path.join(f_model, weights_filename))
        
        optimizer = RMSprop if not self.use_graves else RMSpropGraves
        self.model.compile(loss=loss_func,
                           optimizer=optimizer(lr=self.learning_rate),
                           metrics=['accuracy'])


    def save_model(self, num=None):
        yaml_string = self.model.to_yaml()
        model_name = 'dqn_model.yaml'
        weight_name = 'dqn_model_weights{0}.hdf5'.format((str(num) if num else ''))
        open(os.path.join(f_model, model_name), 'w').write(yaml_string)
        print('save weights')
        self.model.save_weights(os.path.join(f_model, weight_name))

    def end_session(self):
        KTF.set_session(self.old_session)
