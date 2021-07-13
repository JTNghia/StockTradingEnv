# Tutorial by www.pylessons.com
# Tutorial written for - Tensorflow 1.15, Keras 2.2.4

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import random
import gym
import numpy as np
from collections import deque
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.optimizers import Adam, RMSprop
from RL_Bitcoin_trading_bot_1 import *
from datetime import datetime

def DQNModel(input_shape, action_space):
    X_input = Input(input_shape)

    # 'Dense' is the basic form of a neural network layer
    # Input Layer of state size(4) and Hidden Layer with 512 nodes
    X = Dense(512, input_shape=input_shape, activation="relu", kernel_initializer='he_uniform')(X_input)

    # Hidden layer with 256 nodes
    X = Dense(256, activation="relu", kernel_initializer='he_uniform')(X)
    
    # Hidden layer with 64 nodes
    X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)

    # Output Layer with # of actions: 2 nodes (left, right)
    X = Dense(action_space, activation="linear", kernel_initializer='he_uniform')(X)

    model = Model(inputs = X_input, outputs = X, name='CartPole_DQN_model')
    model.compile(loss="mse", optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])

    model.summary()
    return model

class DQNAgent:
    def __init__(self, env, lookback_window_size=50, lr=0.00005, epochs=1, batch_size=200, EPISODES = 10000, gamma = 0.95):
        # df = pd.read_csv('excel_hvn.csv')
        # # df['Date'] = pd.to_datetime(df['Date'])
        # # df = df.sort_values('Date')
        # lookback_window_size = 10
        # num = 10
        # train_df = df[:-num-lookback_window_size]
        # train_df.info()
        # test_df = df[-num-lookback_window_size:] # 30 days


        self.env = env #CustomEnv(df, lookback_window_size=lookback_window_size)
        # by default, CartPole-v1 has max episode steps = 500
        self.state_size = 18*lookback_window_size
        self.action_size = len(self.env.action_space)
        self.EPISODES = EPISODES
        self.memory = deque(maxlen=2000)
        
        self.gamma = gamma    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.999
        self.batch_size = batch_size
        self.train_start = len(env.df)

        # create main model
        self.model = DQNModel(input_shape=(self.state_size,), action_space = self.action_size)
        print(f'env: {self.env}')
        print(f'state_size: {self.state_size}')
        print(f'action_size: {self.action_size}')
        print(f'memory: {self.memory}')

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.train_start:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.model.predict(state))

    def replay(self):
        if len(self.memory) < self.train_start:
            return
        # Randomly sample minibatch from the memory
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))

        state = np.zeros((self.batch_size, self.state_size))
        next_state = np.zeros((self.batch_size, self.state_size))
        action, reward, done = [], [], []

        # do this before prediction
        # for speedup, this could be done on the tensor level
        # but easier to understand using a loop
        for i in range(self.batch_size):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])

        # do batch prediction to save speed
        target = self.model.predict(state)
        target_next = self.model.predict(next_state)

        for i in range(self.batch_size):
            # correction on the Q value for the action used
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                # Standard - DQN
                # DQN chooses the max Q value among next actions
                # selection and evaluation of action is on the target Q Network
                # Q_max = max_a' Q_target(s', a')
                target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))

        # Train the Neural Network with batches
        self.model.fit(state, target, batch_size=self.batch_size, verbose=0)


    def load(self, name):
        self.model = load_model(name)

    def save(self, save_folder = '../weightsDQN', save_filename = ''):
        self.model.save(f'{save_folder}/{save_filename}')
        w = open(f'{save_folder}/DQN_AverageProfits_Log.txt','a+')
        w.write(f'{save_filename}: {np.mean(self.env._profits)}\n')


            
    def train(self, save_folder = '../weightsDQN', save_filename = 'DQN'):
        prev_avg_profits = -99999999
        for e in range(self.EPISODES):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            print('episode: ',e)
            while not done:
                # self.env.render()
                action = self.act(state)
                next_state, reward, done = self.env.step(action)
                if done:     
                    avg_profit = np.mean(self.env._profits)
                    print("episode: {}/{}, avg_profit: {}, e: {:.2}".format(e, self.EPISODES, avg_profit, self.epsilon))
                    # if prev_avg_profits < avg_profit and e > 100:
                    if e >= 100 and avg_profit <= self.env.initial_balance*2 and avg_profit >= self.env.initial_balance:
                        prev_avg_profits = avg_profit
                        print(f"Saving trained model as {save_filename}_Episode({e}).h5")
                        self.save(save_folder=save_folder,save_filename=f'{save_filename}_Episode({e}).h5')
                    # self.env.render_all(f"SaveModel/DQN_Episode({e})")
                    self.env._profits.clear()
                    self.env._networths.clear()
                    break
                next_state = np.reshape(next_state, [1, self.state_size])
                if not done or i == self.env._max_episode_steps-1:
                    reward = reward
                else:
                    reward = -100
                self.remember(state, action, reward, next_state, done)
                state = next_state
                i += 1

                self.replay()

    def retrain(self):
        num = 368
        self.load(f"SaveModel/DQN_Episode({num}).h5")
        prev_avg_profits = -99999999
        for e in range(self.EPISODES):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            print('episode: ',e)
            while not done:
                self.env.render()
                action = self.act(state)
                next_state, reward, done = self.env.step(action)
                if done:     
                    avg_profit = np.mean(self.env._profits)
                    print("episode: {}/{}, avg_profit: {}, e: {:.2}".format(e, self.EPISODES, avg_profit, self.epsilon))
                    if prev_avg_profits < avg_profit:
                        prev_avg_profits = avg_profit
                        print(f"Saving trained model as {save_filename}_Episode({e}).h5")
                        self.save(save_folder=save_folder,save_filename=f'{save_filename}_Episode({e}).h5')
                    # self.env.render_all(f"SaveModel/DQN_Episode({e})")
                    self.env._profits.clear()
                    self.env._networths.clear()
                    break
                next_state = np.reshape(next_state, [1, self.state_size])
                if not done or i == self.env._max_episode_steps-1:
                    reward = reward
                else:
                    reward = -100
                self.remember(state, action, reward, next_state, done)
                state = next_state
                i += 1

                self.replay()


    def test(self, save_folder = '../weightsDQN', filename='DQN_Episode(2).h5', episode = 1):
        self.load(f"{save_folder}/{filename}")
        for e in range(episode):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            while not done:
                # self.env.render()
                action = np.argmax(self.model.predict(state))
                next_state, reward, done= self.env.step(action)
                if done:
                    print("{} episode: {}/{}, avg_profit: {}".format(filename, e, self.EPISODES, np.mean(self.env._profits)))
                    self.env.render_all(f"{filename}.png")
                    return np.mean(self.env._profits)
                    break
                state = np.reshape(next_state, [1, self.state_size])
                i += 1


if __name__ == "__main__":
    df = pd.read_csv('data/fpt_indicators.csv')
    df['Date'] = df['Date'].apply(lambda x: datetime.strptime(str(x),'%Y%m%d'))
    df = df.sort_values('Date')

    lookback_window_size = 10
    test_window = 60 # 60 sessions 
    train_df = df[:-test_window-lookback_window_size]
    test_df = df[-test_window-lookback_window_size:]

    # train_env = CustomEnv(train_df, initial_balance = 1000, lookback_window_size=lookback_window_size)
    # dqn_agent = DQNAgent(train_env, lookback_window_size = lookback_window_size)
    # dqn_agent.train(save_folder='SaveModel/FPT_mean')

    test_env = CustomEnv(test_df, initial_balance = 1000, lookback_window_size=lookback_window_size)
    dqn_agent = DQNAgent(test_env, lookback_window_size = lookback_window_size)
    model_names = os.listdir('SaveModel/FPT_mean')
    avg_profits = []
    avg_profits.append(dqn_agent.test(save_folder='SaveModel/FPT_mean', filename='DQN_Episode(5501).h5'))
    # for filename in model_names:
    #     avg_profits.append(dqn_agent.test(save_folder='SaveModel/FPT_mean', filename=filename))

    # plt.figure(figsize=(30,15)) 
    # plt.plot(model_names, avg_profits)
    # plt.gcf().autofmt_xdate()
    # plt.xlabel('Day',fontsize=20)
    # plt.ylabel('Profit',fontsize=20)
    # plt.legend()
    # plt.show()
    # agent.retrain()
    # agent.test()
