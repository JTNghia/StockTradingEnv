#================================================================
#
#   File name   : RL-Bitcoin-trading-bot_1.py
#   Author      : PyLessons
#   Created date: 2020-12-02
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/RL-Bitcoin-trading-bot
#   Description : Introduction to trading Crypto with Reinforcement Learning
#
#================================================================
import pandas as pd
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

class CustomEnv:
    # A custom Bitcoin trading environment
    def __init__(self, df, initial_balance=1000, lookback_window_size=50):
        # Define action space and state size and other custom parameters
        self.df = df.dropna().reset_index()
        self.df_total_steps = len(self.df)-1
        self.initial_balance = initial_balance
        self.lookback_window_size = lookback_window_size

        # Action space from 0 to 3, 0 is hold, 1 is buy, 2 is sell
        self.action_space = np.array([0, 1, 2])

        # Orders history contains the balance, net_worth, crypto_bought, crypto_sold, crypto_held values for the last lookback_window_size steps
        self.orders_history = deque(maxlen=self.lookback_window_size)
        
        # Market history contains the OHCL values for the last lookback_window_size prices
        self.market_history = deque(maxlen=self.lookback_window_size)

        # State size contains Market+Orders history for the last lookback_window_size steps
        self.state_size = (self.lookback_window_size, 18)

        # For visualization
        self._profits = []
        self._networths = []
        self._dates = []

    # Reset the state of the environment to an initial state
    def reset(self, env_steps_size = 0):
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.crypto_held = 0
        self.crypto_sold = 0
        self.crypto_bought = 0
        self._profits = []
        self._networths = []
        self._dates = []
        if env_steps_size > 0: # used for training dataset
            self.start_step = random.randint(self.lookback_window_size, self.df_total_steps - env_steps_size)
            self.end_step = self.start_step + env_steps_size
        else: # used for testing dataset
            self.start_step = self.lookback_window_size
            self.end_step = self.df_total_steps
            
        self.current_step = self.start_step

        for i in reversed(range(self.lookback_window_size)):
            current_step = self.current_step - i
            self.orders_history.append([self.balance, self.net_worth, self.crypto_bought, self.crypto_sold, self.crypto_held])
            self.market_history.append([self.df.loc[current_step, 'Open'],
                                        self.df.loc[current_step, 'High'],
                                        self.df.loc[current_step, 'Low'],
                                        self.df.loc[current_step, 'Close'],
                                        self.df.loc[current_step, 'Volume'],
                                        self.df.loc[current_step, 'SMA'],
                                        self.df.loc[current_step, 'TENKAN'],
                                        self.df.loc[current_step, 'KIJUN'],
                                        self.df.loc[current_step, 'senkou_span_a'],
                                        self.df.loc[current_step, 'SENKOU'],
                                        self.df.loc[current_step, 'CHIKOU'],
                                        self.df.loc[current_step, 'RSI'],
                                        self.df.loc[current_step, 'EMA']
                                        ])

        state = np.concatenate((self.market_history, self.orders_history), axis=1)
        return state

    # Get the data points for the given current_step
    def _next_observation(self):
        self.market_history.append([self.df.loc[self.current_step, 'Open'],
                                    self.df.loc[self.current_step, 'High'],
                                    self.df.loc[self.current_step, 'Low'],
                                    self.df.loc[self.current_step, 'Close'],
                                    self.df.loc[self.current_step, 'Volume'],
                                    self.df.loc[self.current_step, 'SMA'],
                                    self.df.loc[self.current_step, 'TENKAN'],
                                    self.df.loc[self.current_step, 'KIJUN'],
                                    self.df.loc[self.current_step, 'senkou_span_a'],
                                    self.df.loc[self.current_step, 'SENKOU'],
                                    self.df.loc[self.current_step, 'CHIKOU'],
                                    self.df.loc[self.current_step, 'RSI'],
                                    self.df.loc[self.current_step, 'EMA']
                                    ])
        # print(f'self.market_history: {self.market_history}\n self.orders_history :{self.orders_history}\n\n\n')
        obs = np.concatenate((self.market_history, self.orders_history), axis=1)
        return obs

    # Execute one time step within the environment
    def step(self, action):
        self.crypto_bought = 0
        self.crypto_sold = 0
        self.current_step += 1

        # Set the current price to a random price between open and close
        current_price = random.uniform(
            self.df.loc[self.current_step, 'Open'],
            self.df.loc[self.current_step, 'Close'])
        
        if action == 0: # Hold
            pass
        
        elif action == 1 and self.balance > 0:
            # Buy with 100% of current balance
            self.crypto_bought = self.balance / current_price
            self.balance -= self.crypto_bought * current_price
            self.crypto_held += self.crypto_bought

        elif action == 2 and self.crypto_held>0:
            # Sell 100% of current crypto held
            self.crypto_sold = self.crypto_held
            self.balance += self.crypto_sold * current_price
            self.crypto_held -= self.crypto_sold

        self.prev_net_worth = self.net_worth
        self.net_worth = self.balance + self.crypto_held * current_price

        self.orders_history.append([self.balance, self.net_worth, self.crypto_bought, self.crypto_sold, self.crypto_held])

        # Calculate reward
        reward = self.net_worth - self.prev_net_worth
        self._profits.append(self.net_worth-self.initial_balance)
        self._networths.append(self.net_worth)

        if self.net_worth <= self.initial_balance/2 or self.current_step == self.end_step:
            done = True

        else:
            done = False

        obs = self._next_observation()
        
        return obs, reward, done

    # render environment
    def render(self):
        print(f'Step: {self.current_step}, Net Worth: {self.net_worth}')

    def render_all(self,path_name):
        fontsize = 40
        plt.figure(figsize=(20,13))
        plt.plot(self._profits)
        plt.title(f'Average profit: {round(np.mean(self._profits),3)}',fontsize=fontsize)
        plt.xlabel('Day',fontsize=fontsize)
        plt.ylabel('Profit',fontsize=fontsize)
        # plt.gcf().autofmt_xdate()
        plt.legend()
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.savefig(f'{path_name}')
        plt.show()

        # plt.savefig(f'{path_name}_Average_profit.png')


        # plt.figure(figsize=(30,15))
        # plt.plot(self._networths)
        # plt.title(f'Average networths: {np.mean(self._networths)}',fontsize=20)
        # plt.xlabel('Day',fontsize=20)
        # plt.ylabel('Profit',fontsize=20)
        # plt.gcf().autofmt_xdate()
        # plt.legend()
        # plt.savefig(f'{path_name}_Average_networths.png')

        # w = open('DQN_HVN_AverageProfits_Log.txt','a+')
        # w.write(f'{path_name}: {np.mean(self._profits)}\n')
        # wn = open('DQN_HVN_AverageNetworths_Log.txt','a+')
        # wn.write(f'{path_name}: {np.mean(self._networths)}\n')

        
def Random_games(env, train_episodes = 50, training_batch_size=500):
    average_net_worth = 0
    for episode in range(train_episodes):
        state = env.reset(env_steps_size = training_batch_size)

        while True:
            # env.render()

            action = np.random.randint(3, size=1)[0]
            pre_state = state
            state, reward, done = env.step(action)
            # print(state.shape)

            if env.current_step == env.end_step:
                average_net_worth += env.net_worth
                print("net_worth:", env.net_worth)
                break
        break

    print("average_net_worth:", average_net_worth/train_episodes)


# df = pd.read_csv('pricedata.csv')
# df = df.sort_values('Date')

# lookback_window_size = 10
# train_df = df[:-720-lookback_window_size]
# test_df = df[-720-lookback_window_size:] # 30 days

# train_env = CustomEnv(train_df, lookback_window_size=lookback_window_size)
# test_env = CustomEnv(test_df, lookback_window_size=lookback_window_size)

# Random_games(train_env, train_episodes = 10, training_batch_size=500)
