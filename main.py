from __future__ import absolute_import, division, print_function, unicode_literals
import functools
import numpy as np
import tensorflow as tf
import pandas as pd
import math
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from collections import deque, namedtuple
from time import time
from numpy.random import random, randint, seed
from random import sample
from matplotlib.ticker import FuncFormatter
import gc
import resource


class MyModel(tf.keras.Model):
    def __init__(self, num_states, hidden_units, num_actions):
        super(MyModel, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(num_states,))
        self.hidden_layers = []
        for i in hidden_units:
            self.hidden_layers.append(tf.keras.layers.Dense(
                i, activation='relu', kernel_initializer='RandomUniform', kernel_regularizer=tf.keras.regularizers.l2(.000001)))
        self.output_layer = tf.keras.layers.Dense(num_actions, activation='linear')

    @tf.function
    def call(self, inputs):
        z = self.input_layer(inputs)
        for layer in self.hidden_layers:
            z = layer(z)
        output = self.output_layer(z)
        return output


class DQN:
    def __init__(self, num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr):
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.optimizer = tf.optimizers.Adam(lr)
        self.gamma = gamma
        self.model = MyModel(num_states, hidden_units, num_actions)
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences

    def predict(self, inputs):
        return self.model(np.atleast_2d(inputs.astype('float32')))

    @tf.function
    def train(self, TargetNet):
        if len(self.experience['s']) < self.min_experiences:
            return 0
        ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)
        states = np.asarray([self.experience['s'][i] for i in ids])
        actions = np.asarray([self.experience['a'][i] for i in ids])
        rewards = np.asarray([self.experience['r'][i] for i in ids])
        states_next = np.asarray([self.experience['s2'][i] for i in ids])
        dones = np.asarray([self.experience['done'][i] for i in ids])
        value_next = np.max(TargetNet.predict(states_next), axis=1)
        actual_values = np.where(dones, rewards, rewards + self.gamma * value_next)

        with tf.GradientTape() as tape:
            selected_action_values = self.predict(states)
            loss = tf.keras.losses.MSE(actual_values, selected_action_values)
        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

    def get_action(self, states, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.predict(np.atleast_2d(states))[0])

    def add_experience(self, exp):
        if len(self.experience['s']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        for key, value in exp.items():
            self.experience[key].append(value)

    def copy_weights(self, TrainNet):
        variables1 = self.model.trainable_variables
        variables2 = TrainNet.model.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())


def momentum100(data, window=100):
    def pct_rank(x):
        return pd.Series(x).rank(pct=True).iloc[-1]

    return data.rolling(window).apply(pct_rank, raw=True)


def momentum20(data, window=20):
    def pct_rank(x):
        return pd.Series(x).rank(pct=True).iloc[-1]

    return data.rolling(window).apply(pct_rank, raw=True)


def rsi(data, window=14):
    diff = data.diff().dropna()

    up, down = diff.copy(), diff.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    rolling_up = up.rolling(window).mean()
    rolling_down = down.abs().rolling(window).mean()

    RS2 = rolling_up / rolling_down
    return 100 - (100 / (1 + RS2))

def process_train(path):
    parse_dates = ['Date']
    train_df = pd.read_csv(path, parse_dates=parse_dates)
    train = train_df[['Date', 'Adj Close', 'Volume']].dropna()
    train.columns = ['date', 'close', 'volume']

    train['returns'] = train['close'].pct_change()
    train['close_pct_100'] = momentum100(train['close'])
    train['volume_pct_100'] = momentum100(train['volume'])
    train['close_pct_20'] = momentum20(train['close'])
    train['volume_pct_20'] = momentum20(train['volume'])
    train['return_5'] = train['returns'].pct_change(5)
    train['return_21'] = train['returns'].pct_change(21)
    train['rsi'] = rsi(train['close'])
    train = train.replace((np.inf, -np.inf), np.nan).dropna()
    #train = train.loc[(train['date'] < '2011-06-01')]
    train = train.iloc[:, 1:]
    return train

def main():

    data = process_train('BRK-B.csv')

    num_states = 13
    num_actions = 3
    hidden_units = [256, 256, 256]
    gamma = 0.99
    max_experiences = 480
    min_experiences = 100
    minibatch_size = 96
    learning_rate = .00025
    tau = 100
    epsilon = 1
    decay = 0.9999


    TrainNet = DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, minibatch_size,
                   learning_rate)
    TargetNet = DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, minibatch_size,
                    learning_rate)

    steps = 63
    max_episodes = 10000
    numberOfRows = steps*max_episodes
    trading_cost_bps = 1e-3
    time_cost_bps = 1e-4

    trade_log = pd.DataFrame(index=np.arange(0, numberOfRows), columns=("Price", "Action", "Stock Holdings",
                                                                        "Portfolio Value", "Buy and Hold Value"))

    episodes = []
    advantages = []

    for episode in range(max_episodes):

        episodes.append(episode)
        cash = 100000
        stock_holdings = 100000
        buy_and_hold = 100000
        buy_and_hold_cash = 100000
        trade_units = 1000
        portfolio_value = 100000
        market_value = 100000

        done = False

        high = len(data.index) - steps
        start_index = np.random.randint(low=0, high=high)
        actions = []

        for step in range(steps):

            advs = []

            index = start_index + step
            observation = data.iloc[index]
            price = observation[0]
            market_return = observation[2]

            if step == 0:
                action = 1
                portfolio_value = (price * stock_holdings) + cash
                market_value = (price * buy_and_hold) + buy_and_hold_cash

                portfolio = np.array([stock_holdings, portfolio_value, market_value])
                observation = np.append(observation, portfolio)

                trade_log.loc[((episode*steps)+step)] = [price, 1, stock_holdings, portfolio_value, market_value]

            else:
                if epsilon > .01:
                    epsilon -= 1.8e-8

                portfolio = np.array([stock_holdings, portfolio_value, market_value])
                observation = np.append(observation, portfolio)

                action = TrainNet.get_action(observation, epsilon)
                action = action - 1
                prev_observation = observation

                observation = data.iloc[index + 1]
                price = observation[0]
                market_return = observation[2]

                new_market_value = (price * buy_and_hold) + buy_and_hold_cash #- time_cost_bps
                new_portfolio_value = portfolio_value

                if action == 0:
                    new_portfolio_value = (stock_holdings * price) + cash #- time_cost_bps
                elif stock_holdings < trade_units and action == -1:
                    action = 0
                    new_portfolio_value = (stock_holdings * price) + cash #- time_cost_bps
                elif stock_holdings >= trade_units and action == -1:
                    stock_holdings -= trade_units
                    cash += (price * trade_units) #- ((trading_cost_bps + time_cost_bps) * trade_units)
                    new_portfolio_value = (stock_holdings * price) + cash
                elif cash >= (price*trade_units) and action == 1:
                    stock_holdings += trade_units
                    cash -= (price * trade_units) #+ ((trading_cost_bps + time_cost_bps) * trade_units)
                    new_portfolio_value = (stock_holdings * price) + cash
                else:
                    new_portfolio_value = (stock_holdings * price) + cash #- time_cost_bps

                reward = action * market_return
                #reward = math.log(new_portfolio_value/portfolio_value)
                #reward = (new_portfolio_value/new_market_value)

                portfolio = np.array([stock_holdings, new_portfolio_value, new_market_value])
                observation = np.append(observation, portfolio)

                trade_log.loc[((episode * steps) + step)] = [price, action, stock_holdings, new_portfolio_value, new_market_value]

                portfolio_value = new_portfolio_value
                market_value = new_market_value

                exp = {'s': prev_observation, 'a': action, 'r': reward, 's2': observation, 'done': done}
                TrainNet.add_experience(exp)
                TrainNet.train(TargetNet)

                if step % tau == 0:
                    TargetNet.copy_weights(TrainNet)

            advs.append(((portfolio_value / market_value) - 1))
            actions.append(action)

            if epsilon > .01:
                epsilon = epsilon * decay

        advantages.append(np.mean(advs))
        print("Sell: ", actions.count(-1), " Hold: ", actions.count(0), " Buy: ", actions.count(1))
        print("Cash: ", cash)
        print("Price: ", price)


        if episode % 1 == 0:
            print('Episode: {:5d} | Epsilon: {:1.3f} | '
                  'Agent Stocks : {:7d} | Total Agent Value: {:09.2f} | Buy and Hold Value: {:09.2f} | Agent Advantage %: {:+2.2f}'.format(
                episode, epsilon, stock_holdings, portfolio_value,  market_value, np.mean(advs)*100))


    e = []
    a = []
    for i in episodes:
        if i%50==0:
            e.append(i)
    for i in range(len(advantages)):
        if i%50==0:
            a.append(advantages[i])
    plt.suptitle('Berkshire Hathaway', fontsize=20)
    plt.plot(e, a)
    plt.grid()
    plt.show()

main()