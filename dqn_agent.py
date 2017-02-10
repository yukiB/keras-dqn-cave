from collections import deque
import os
import random
import copy
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import InputLayer, Convolution2D
from keras.models import model_from_yaml
import keras.callbacks
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf

f_log = './log'
f_model = './models'

model_filename = 'dqn_model.yaml'
weights_filename = 'dqn_model_weights.hdf5'
    


class DQNAgent:
    """
    Multi Layer Perceptron with Experience Replay
    """

    def __init__(self, enable_actions, env_name, env_size):
        # parameters
        self.name = os.path.splitext(os.path.basename(__file__))[0]
        self.environment_name = env_name
        self.env_size = env_size
        self.enable_actions = enable_actions
        self.n_actions = len(self.enable_actions)
        self.minibatch_size = 32
        self.replay_memory_size = 10000
        self.learning_rate = 0.00001
        self.momentum = 0.95
        self.min_grad = 0.01
        self.discount_factor = 0.9
        self.exploration = 0.1
        self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        self.model_name = "{}.ckpt".format(self.environment_name)

        self.old_session = KTF.get_session()
        self.session = tf.Session('')
        KTF.set_session(self.session)

        # replay memory
        self.D = deque(maxlen=self.replay_memory_size)

        # state
        self.state_num = 4

        # variables
        self.current_loss = 0.0


    def init_model(self, model=None):
        def _create_placeholder(_model):
            s = tf.placeholder(tf.float32, [None, self.state_num, self.env_size[0], self.env_size[1]])
            q_values = _model(s)
            return s, q_values, _model

        if model == None:
            model = self.create_model()
            init = False
        else:
            init = True
        # Q Networkの構築
        self.s, self.q_values, self.q_network = _create_placeholder(copy.copy(model))
        q_network_weights = self.q_network.trainable_weights

        # Target Networkの構築
        self.st, self.target_q_values, self.target_network = _create_placeholder(copy.copy(model))
        target_network_weights = self.target_network.trainable_weights

        # 定期的にTarget Networkを更新するための処理の構築
        self.update_target_network = [target_network_weights[i].assign(q_network_weights[i]) for i in range(len(target_network_weights))]

        # 誤差関数や最適化のための処理の構築
        self.a, self.y, self.loss, self.training = self.training_op(q_network_weights)

        # Sessionの構築
        self.sess = tf.InteractiveSession()
        
        
        self.sess.run(tf.global_variables_initializer())

        # Target Networkの初期化
        self.sess.run(self.update_target_network)

        

    def create_model(self):

        model = Sequential()
        print(self.env_size)
        model.add(InputLayer(input_shape=(self.state_num, *self.env_size)))
        model.add(Convolution2D(32, 8, 8, border_mode='same', activation='relu', subsample=(4, 4)))
        model.add(Convolution2D(64, 4, 4, border_mode='same', activation='relu', subsample=(2, 2)))
        model.add(Convolution2D(64, 4, 4, border_mode='same', activation='relu', subsample=(2, 2)))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.n_actions))
        #self.model.compile(loss='mean_squared_error',
        #                   optimizer="rmsprop",
        #                   metrics=['accuracy'])
        return model


    def training_op(self, q_network_weights):
        a = tf.placeholder(tf.int64, [None])
        y = tf.placeholder(tf.float32, [None])

        a_one_hot = tf.one_hot(a, self.n_actions, 1.0, 0.0)
        q_value = tf.reduce_sum(tf.mul(self.q_values, a_one_hot), reduction_indices=1)

        error = tf.abs(y - q_value)
        quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part)

        optimizer = tf.train.RMSPropOptimizer(self.learning_rate, momentum=self.momentum, epsilon=self.min_grad)
        training = optimizer.minimize(loss, var_list=q_network_weights)

        return a, y, loss, training
    

    def Q_values(self, states):
        # Q(state, action) of all actions
        res = self.q_network.predict(np.array([states]))
        #self.q_values.eval(feed_dict={self.s: [np.float32(state / 255.0)]})
        return res[0]

    def select_action(self, states, epsilon):
        if np.random.rand() <= epsilon:
            # random
            return np.random.choice(self.enable_actions)
        else:
            # max_action Q(state, action)
            return self.enable_actions[np.argmax(self.Q_values(states))]

    def store_experience(self, states, action, reward, state_1, terminal):
        self.D.append((states, action, reward, state_1, terminal))
        if len(self.D) < self.replay_memory_size:
            return False
        else:
            return True

    def experience_replay(self):
        state_minibatch = []
        action_minibatch = []
        y_minibatch = []

        # sample random minibatch
        minibatch_size = min(len(self.D), self.minibatch_size)
        minibatch_indexes = np.random.randint(0, len(self.D), minibatch_size)

        for j in minibatch_indexes:
            state_j, action_j, reward_j, state_j_1, terminal = self.D[j]
            action_j_index = self.enable_actions.index(action_j)

            y_j = self.Q_values(state_j)

            if terminal:
                y_j[action_j_index] = reward_j
            else:
                # reward_j + gamma * max_action' Q(state', action') alpha(learing rate) = 1
                y_j[action_j_index] = reward_j + self.discount_factor * np.max(self.Q_values(state_j_1))  # NOQA

            state_minibatch.append(state_j)
            y_minibatch.append(y_j[action_j_index])
            action_minibatch.append(action_j_index)

        # training
        #self.model.fit(np.array(state_minibatch), np.array(y_minibatch), verbose=0)
        loss, _ = self.sess.run([self.loss, self.training],
                                feed_dict={
                                    self.s: np.array(state_minibatch),
                                    self.a: np.array(action_minibatch),
                                    self.y: np.array(y_minibatch)
                                })

        # for log
        #score = self.model.evaluate(np.array(state_minibatch), np.array(y_minibatch), verbose=0)
        self.current_loss = loss

    def load_model2(self, model_path=None):

        yaml_string = open(os.path.join(f_model, model_filename)).read()
        model = model_from_yaml(yaml_string)
        model.load_weights(os.path.join(f_model, weights_filename))
        self.init_model(model)

        
    def load_model(self, model_path=None):
        checkpoint = tf.train.get_checkpoint_state('./model/test')
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print('Successfully loaded: ' + checkpoint.model_checkpoint_path)
        else:
            print('Training new network...')
        

    def save_model(self, num=None, t=None):
        #yaml_string = self.q_network.to_yaml()
        #model_name = 'dqn_model{0}.yaml'.format((str(num) if num else ''))
        #weight_name = 'dqn_model_weights{0}.hdf5'.format((str(num) if num else ''))
        #open(os.path.join(f_model, model_name), 'w').write(yaml_string)
        #print('save weights')
        #self.q_network.save_weights(os.path.join(f_model, weight_name))

        save_path = self.saver.save(self.sess, './model/test', global_step=(t))        

    def end_session(self):
        KTF.set_session(self.old_session)
