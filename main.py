from __future__ import absolute_import, division, print_function, unicode_literals
import functools
import numpy as np
import tensorflow as tf
import pandas as pd
from math import log
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from collections import deque, namedtuple
from time import time
from numpy.random import random, randint, seed
from random import sample
from matplotlib.ticker import FuncFormatter

"""
IMPORT AND PROCESS DATA
"""


class Data:

    def __init__(self, d, trading_days):
        self.trading_days = trading_days + 1
        self.min_perc_days = 100
        self.data = d
        self.min_values = self.data.min()
        self.max_values = self.data.max()
        self.step = 0
        self.index = 0

    def reset(self):
        high = len(self.data.index) - self.trading_days
        self.index = np.random.randint(low=0, high=high)
        self.step = 0
        return

    def take_step(self):
        obs = self.data.iloc[self.index].values
        self.index += 1
        self.step += 1
        done = self.step >= self.trading_days
        return obs, done


class Simulator:

    def __init__(self, steps, trading_cost_bps, time_cost_bps):
        self.trading_cost_bps = trading_cost_bps
        self.time_cost_bps = time_cost_bps
        self.steps = steps
        self.step = 0
        self.actions = np.zeros(self.steps)
        self.navs = np.ones(self.steps)
        self.market_navs = np.ones(self.steps)
        self.strategy_returns = np.ones(self.steps)
        self.positions = np.zeros(self.steps)
        self.costs = np.zeros(self.steps)
        self.trades = np.zeros(self.steps)
        self.market_returns = np.zeros(self.steps)

    def reset(self):
        self.step = 0
        self.actions.fill(0)
        self.navs.fill(1)
        self.market_navs.fill(1)
        self.strategy_returns.fill(0)
        self.positions.fill(0)
        self.costs.fill(0)
        self.trades.fill(0)
        self.market_returns.fill(0)

    def take_step(self, value, action, market_return):
        bod_position = 0.0 if self.step == 0 else self.positions[self.step - 1]
        bod_nav = 1.0 if self.step == 0 else self.navs[self.step - 1]
        bod_market_nav = 1.0 if self.step == 0 else self.market_navs[self.step - 1]

        self.market_returns[self.step] = market_return
        self.actions[self.step] = action

        self.positions[self.step] = action - 1

        self.trades[self.step] = self.positions[self.step] - bod_position

        trade_costs_pct = abs(self.trades[self.step]) * self.trading_cost_bps
        self.costs[self.step] = trade_costs_pct + self.time_cost_bps
        reward = ((bod_position * market_return) - self.costs[self.step])
        val_change = (bod_position * market_return) - self.costs[self.step]
        #reward = log((val_change + value)/value)
        self.strategy_returns[self.step] = reward

        if self.step != 0:
            self.navs[self.step] = bod_nav * (1 + self.strategy_returns[self.step - 1])
            self.market_navs[self.step] = bod_market_nav * (1 + self.market_returns[self.step - 1])

        nav = self.navs[self.step]
        market_nav = self.market_navs[self.step]
        self.step += 1
        return reward, nav, market_return, val_change

    def result(self):
        return pd.DataFrame({'action'         : self.actions,  # current action
                             'nav'            : self.navs,  # starting Net Asset Value (NAV)
                             'market_nav'     : self.market_navs,
                             'market_return'  : self.market_returns,
                             'strategy_return': self.strategy_returns,
                             'position'       : self.positions,  # eod position
                             'cost'           : self.costs,  # eod costs
                             'trade'          : self.trades})  # eod trade)


class TradingEnvironment():

    def __init__(self, data, trading_days=252, trading_cost_bps=1e-3, time_cost_bps=1e-4):
        self.trading_days = trading_days
        self.trading_cost_bps = trading_cost_bps
        self.time_cost_bps = time_cost_bps
        self.src = Data(d=data, trading_days=self.trading_days)
        self.sim = Simulator(steps=self.trading_days, trading_cost_bps=self.trading_cost_bps, time_cost_bps=self.time_cost_bps)
        self.reset()

    def step(self, action, value):
        observation, done = self.src.take_step()
        reward, nav, market_return, val_change = self.sim.take_step(value, action=action,
                                          market_return=observation[2])
        return observation, reward, done, nav, market_return, val_change

    def reset(self):
        self.src.reset()
        self.sim.reset()
        return self.src.take_step()[0]

    def get_results(self):
        return self.sim.result()

    def run_strategy(self, strategy, return_df=True):
        observation = self.reset()
        done = False
        while not done:
            action = strategy(observation, self)
            observation, reward, done, info = self.step(action)

        return self.sim.result() if return_df else None


"""
DEFINE NEURAL NETWORKS
"""


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
            #tf.nn.sigmoid_cross_entropy_with_logits
            #tf.keras.losses.MSE
            loss = tf.keras.losses.MSE(actual_values, selected_action_values)
        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

    def get_action(self, states, epsilon):
        # if np.random.random() < epsilon:
        #     return np.random.choice(self.num_actions)
        # else:
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


"""
TRADING FUNCTION
"""


def trade(data, TrainNet, TargetNet, epsilon, copy_step, val):
    rewards_total = 0
    value = val
    market_value = val
    rewards = []
    step = 0
    done = False
    env = TradingEnvironment(data)
    observation = TradingEnvironment.reset(env)
    navs, market_navs, diffs = [], [], []
    while not done:
        if epsilon > .01:
            epsilon -= 1.8e-6
        action = TrainNet.get_action(observation, epsilon)
        prev_observation = observation
        observation, reward, done, nav, market_reward, val_change = TradingEnvironment.step(env, action, value)

        reward_n = 1
        last_r = 1
        if len(rewards) != 0:
            reward_n = np.sum(rewards)
            last_r = rewards[-1]

        r = (1 + (action - 1)*((reward-last_r)/last_r))*(last_r/reward_n)

        value = value + val_change
        market_value = market_value + market_reward
        rewards.append(reward)
        navs.append(nav)
        market_navs.append(market_reward)
        diff = nav - market_reward
        diffs.append(diff)
        rewards_total += reward
        if done:
            _ = TradingEnvironment.reset(env)

        a2, a3 = 0, 0
        r2, r3 = 0, 0
        if action == 0:
            a2 = 1
            a3 = 2
            r2 = r * 0
            r3 = r * -1
        elif action == 1:
            a2 = 0
            a3 = 2
            r2 = (1 + (a2 - 1)*((reward-last_r)/last_r))*(last_r/reward_n)
            r3 = (1 + (a3 - 1)*((reward-last_r)/last_r))*(last_r/reward_n)
        elif action == 2:
            a2 = 0
            a3 = 1
            r2 = (1 + (a2 - 1)*((reward-last_r)/last_r))*(last_r/reward_n)
            r3 = (1 + (a3 - 1)*((reward-last_r)/last_r))*(last_r/reward_n)

        exp = {'s': prev_observation, 'a': action, 'r': r, 's2': observation, 'done': done}
        exp2 = {'s': prev_observation, 'a': a2, 'r': r2, 's2': observation, 'done': done}
        exp3 = {'s': prev_observation, 'a': a3, 'r': r3, 's2': observation, 'done': done}
        TrainNet.add_experience(exp)
        TrainNet.add_experience(exp2)
        TrainNet.add_experience(exp3)


        if step % 5 == 0:
            TrainNet.train(TargetNet)

        step += 1
        if step % copy_step == 0:
            TargetNet.copy_weights(TrainNet)

    return rewards_total, np.mean(navs), market_value, np.mean(diffs), epsilon, value


"""
PROCESS DATA
"""


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


def process(path):
        parse_dates = ['date']
        train_df = pd.read_csv(path, parse_dates=parse_dates)
        train = train_df[['date', 'adj_close', 'adj_volume']].dropna()
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
        r = train['returns'].copy()
        d = train['date'].copy()
        scaler = StandardScaler()
        train[['close', 'volume', 'close_pct_100', 'volume_pct_100', 'close_pct_20', 'volume_pct_20', 'return_5',
               'return_21', 'rsi']] = scaler.fit_transform(
                train[['close', 'volume', 'close_pct_100', 'volume_pct_100', 'close_pct_20', 'volume_pct_20', 'return_5', 'return_21', 'rsi']])

        test = train.loc[(train['date'] >= '2011-06-01')]
        test = test.iloc[:, 1:]
        train = train.loc[(train['date'] < '2011-06-01')]
        train = train.iloc[:, 1:]
        return train

def process_val(path):
    parse_dates = ['date']
    train_df = pd.read_csv(path, parse_dates=parse_dates)
    train = train_df[['date', 'adj_close', 'adj_volume']].dropna()
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
    r = train['returns'].copy()
    d = train['date'].copy()
    scaler = StandardScaler()
    train[['close', 'volume', 'close_pct_100', 'volume_pct_100', 'close_pct_20', 'volume_pct_20', 'return_5',
           'return_21', 'rsi']] = scaler.fit_transform(
        train[['close', 'volume', 'close_pct_100', 'volume_pct_100', 'close_pct_20', 'volume_pct_20', 'return_5',
               'return_21', 'rsi']])

    test = train.loc[(train['date'] >= '2011-06-01')]
    test = test.iloc[:, 1:]
    train = train.loc[(train['date'] < '2011-06-01')]
    train = train.iloc[:, 1:]
    return test

class Val_TradingEnvironment():

    def __init__(self, data, trading_days, trading_cost_bps=1e-3, time_cost_bps=1e-4):
        self.trading_days = trading_days
        self.trading_cost_bps = trading_cost_bps
        self.time_cost_bps = time_cost_bps
        self.src = Data(d=data, trading_days=self.trading_days)
        self.sim = Simulator(steps=self.trading_days, trading_cost_bps=self.trading_cost_bps, time_cost_bps=self.time_cost_bps)


    def step(self, action, observation):
        reward, nav, market_nav = self.sim.take_step(action=action,
                                          market_return=observation[2])
        return reward, nav, market_nav


def main():

    # PROCESS DATA
    data = process('train_prices.csv')

    # DECLARE VARIABLES
    num_states = 10
    num_actions = 3
    hidden_units = [256, 256, 256]
    gamma = 0.99
    max_experiences = 480
    min_experiences = 100
    minibatch_size = 96
    learning_rate = .00025

    # DEFINE NETWORKS

    TrainNet = DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, minibatch_size, learning_rate)
    TargetNet = DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, minibatch_size, learning_rate)

    # DEFINE TRAINING VARIABLES
    max_episodes = 10000
    epsilon = 1
    decay = 0.9999
    copy_step = 100
    episodes = []
    average_rewards = []
    navs, market_navs, diffs = [], [], []
    total_rewards = np.empty(max_episodes)
    runtime = 0
    total_value = 10000

    for episode in range(max_episodes):
        episode_start = time()

        total_reward, nav, market_nav, diff, eps, value = trade(data, TrainNet, TargetNet, epsilon, copy_step, total_value)

        total_value = value
        total_rewards[episode] += total_reward
        navs.append(nav)
        market_navs.append(market_nav)
        diffs.append(diff)
        avg_rewards = total_rewards[max(0, episode - 100):(episode + 1)].mean()
        average_rewards.append(avg_rewards)
        episodes.append(episode)
        episode_time = time() - episode_start
        runtime += episode_time

        if eps > .01:
            epsilon = eps * decay

        # if episode % 100 == 0:
        #     print('Episode: {:>4d} | Value: {:>6.3f} | Avg Reward: {:>5.3f} | '
        #           'NAV: {:>5.3f} | Market Value: {:>5.3f} | Delta: {:4.0f}'.format(episode,
        #             value, avg_rewards, navs[episode], market_navs[episode], np.sum([s > 0 for s in diffs[-100:]])))

        if episode % 100 == 0:
            print('Episode: {:>4d} | Value: {:>6.3f} | Avg Reward: {:>5.3f} | Market Value: {:>5.3f}'.format(episode,
                    value, avg_rewards, market_navs[episode]))


    results = pd.DataFrame({'episode': list(range(1, episode + 2)),
                            'nav': navs,
                            'market_nav': market_navs,
                            'outperform': diffs})

    fn = 'trading_agent_result_no_cost.csv'
    results.to_csv(fn, index=False)

    results = pd.read_csv('trading_agent_result_no_cost.csv')
    results.columns = ['Episode', 'Agent', 'Market', 'difference']
    results = results.set_index('Episode')
    results['Strategy Wins (%)'] = (results.difference > 0).rolling(100).sum()
    results.info()

    fig, axes = plt.subplots(ncols=2, figsize=(14,4))
    (results[['Agent', 'Market']]
        .sub(1)
        .rolling(100)
        .mean()
        .plot(ax=axes[0],
            title='Annual Returns (Moving Average)', lw=1))
    results['Strategy Wins (%)'].div(100).rolling(50).mean().plot(ax=axes[1], title='Agent Outperformance (%, Moving Average)');
    for ax in axes:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:,.0f}'.format(x)))
    fig.tight_layout()
    fig.savefig('trading_agent', dpi=300)


    val_data = process_val('train_prices.csv')
    days = len(val_data.index)
    env = Val_TradingEnvironment(val_data, days)
    steps, val_rewards, val_navs, val_market_navs = [], [], [], []
    for i in range(days):
        epsilon = 0
        observation = val_data.iloc[i]
        action = TrainNet.get_action(observation, epsilon)
        reward, nav, market_nav = Val_TradingEnvironment.step(env, action, observation)
        val_rewards.append(reward)
        val_navs.append(nav)
        val_market_navs.append(market_nav)
        steps.append(i)

    print("Sum agent rewards: ", sum(val_rewards))
    print("Sum agent returns: ", sum(val_navs))
    print("Sum market returns: ", sum(val_market_navs))


main()