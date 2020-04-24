from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
import pandas as pd
import math
from tensorflow import keras
import matplotlib.pyplot as plt
from scipy import signal


class MyModel(tf.keras.Model):
    def __init__(self, num_states, hidden_units, num_actions):
        super(MyModel, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(num_states,))
        self.hidden_layers = []
        for i in hidden_units:
            self.hidden_layers.append(tf.keras.layers.Dense(
                i, activation='linear', kernel_initializer='RandomUniform'))
        self.output_layer = tf.keras.layers.Dense(num_actions, activation='linear', kernel_initializer='RandomUniform')

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

    def train(self, TargetNet):
        if len(self.experience['s']) < self.min_experiences:
            return 0
        else:
            ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)
            states = np.asarray([self.experience['s'][i] for i in ids])
            actions = np.asarray([self.experience['a'][i] for i in ids])
            rewards = np.asarray([self.experience['r'][i] for i in ids])
            states_next = np.asarray([self.experience['s2'][i] for i in ids])
            dones = np.asarray([self.experience['done'][i] for i in ids])
            value_next = np.max(TargetNet.predict(states_next), axis=1)
            actual_values = np.where(dones, rewards, rewards + self.gamma * value_next)

            with tf.GradientTape() as tape:
                selected_action_values = tf.math.reduce_sum(
                    self.predict(states) * tf.one_hot(actions, self.num_actions), axis=1)
                loss = tf.math.reduce_sum(tf.square(actual_values - selected_action_values))


            variables = self.model.trainable_variables
            gradients = tape.gradient(loss, variables)
            self.optimizer.apply_gradients(zip(gradients, variables))

    def eva(self, data, index, k):
        predictions = []
        for i in range(1, k + 1):
            predictions.append(self.predict(np.atleast_2d(data.iloc[index - k]))[0])
        action_a = []
        action_b = []
        action_c = []
        for i in predictions:
            action_a.append(i[0])
            action_b.append(i[1])
            action_c.append(i[2])
        avg_a = np.mean(action_a)
        avg_b = np.mean(action_b)
        avg_c = np.mean(action_c)
        p = [avg_a, avg_b, avg_c]
        return p

    def get_action_eva(self, states, epsilon):
        if np.random.random() < epsilon:
            return [-999, -999, -999]
        else:
            return self.predict(np.atleast_2d(states))

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

def process_train(path, split_date):
    parse_dates = ['Date']
    train_df = pd.read_csv(path, parse_dates=parse_dates)
    train = train_df[['Date', 'Adj Close', 'Volume']].dropna()
    train.columns = ['date', 'close', 'volume']
    detrend_close = signal.detrend(train['close'])
    detrend_close += 25

    adj_close = train['close']

    train['returns'] = adj_close.pct_change()
    train['close_pct_100'] = momentum100(adj_close)
    train['volume_pct_100'] = momentum100(train['volume'])
    train['close_pct_20'] = momentum20(adj_close)
    train['volume_pct_20'] = momentum20(train['volume'])
    train['return_5'] = train['returns'].pct_change(5)
    train['return_21'] = train['returns'].pct_change(21)
    train['rsi'] = rsi(adj_close)
    train = train.replace((np.inf, -np.inf), np.nan).dropna()
    train = train.loc[(train['date'] < split_date)]
    train = train.iloc[:, 1:]
    return train

def process_test(path, split_date):
    parse_dates = ['Date']
    train_df = pd.read_csv(path, parse_dates=parse_dates)
    train = train_df[['Date', 'Adj Close', 'Volume']].dropna()
    train.columns = ['date', 'close', 'volume']
    detrend_close = signal.detrend(train['close'])
    detrend_close += 25

    adj_close = train['close']

    train['returns'] = adj_close.pct_change()
    train['close_pct_100'] = momentum100(adj_close)
    train['volume_pct_100'] = momentum100(train['volume'])
    train['close_pct_20'] = momentum20(adj_close)
    train['volume_pct_20'] = momentum20(train['volume'])
    train['return_5'] = train['returns'].pct_change(5)
    train['return_21'] = train['returns'].pct_change(21)
    train['rsi'] = rsi(adj_close)
    train = train.replace((np.inf, -np.inf), np.nan).dropna()
    train = train.loc[(train['date'] >= split_date)]
    train = train.iloc[:, 1:]
    return train

def main():

    stock = 'BP'
    split_date = '2019-01-01'
    if stock in ['MNST', 'APA', 'AAPL']:
        split_date = '2017-01-01'

    path = stock + '.csv'
    data = process_train(path, split_date)

    test_data = process_test(path, split_date)

    num_states = 13
    num_actions = 3
    hidden_units = [256, 256, 256, 256, 256]
    gamma = 0.99
    max_experiences = 10000
    min_experiences = 100
    minibatch_size = 32
    learning_rate = 1e-3
    tau = 25
    epsilon = 1

    TrainNet = DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, minibatch_size,
                   learning_rate)
    TargetNet = DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, minibatch_size,
                    learning_rate)

    steps = 63
    max_episodes = 500
    numberOfRows = steps*max_episodes

    trade_log = pd.DataFrame(index=np.arange(0, numberOfRows), columns=("Price", "Action", "Stock Holdings",
                                                                        "Portfolio Value", "Buy and Hold Value"))

    state_log = pd.DataFrame(index=np.arange(0, numberOfRows), columns=("Close", "Volume", "Returns", "Close_100",
                                                                        "Volume_100", "Close_20", "Volume_20",
                                                                        "Returns_5", "Returns_21", "RSI",
                                                                        "Holdings", "Portfolio_Val", "Market_Val"))

    episodes = []
    advantages = []

    for episode in range(max_episodes):

        episodes.append(episode)
        cash = 1000
        stock_holdings = 100
        buy_and_hold = 100
        buy_and_hold_cash = 1000
        trade_units = 1
        portfolio_value = 1000
        market_value = 1000
        k = 10

        done = False

        high = len(data.index) - (steps) - 1
        start_index = np.random.randint(low=0, high=high)
        actions = []

        rewards = []

        for step in range(steps):

            advs = []

            index = start_index + step
            observation = data.iloc[index]
            price = observation[0]
            market_return = observation[2]

            if step == 0:
                action = 1
                print("Start Price: ", price)
                portfolio_value = (price * stock_holdings) + cash
                market_value = (price * buy_and_hold) + buy_and_hold_cash

                portfolio = np.array([stock_holdings, portfolio_value, market_value])
                observation = np.append(observation, portfolio)
                trade_log.loc[((episode*steps)+step)] = [price, 1, stock_holdings, portfolio_value, market_value]

            else:

                portfolio = np.array([stock_holdings, portfolio_value, market_value])
                observation = np.append(observation, portfolio)
                eva_index = (episode * steps) + step
                state_log.loc[eva_index] = observation

                action = TrainNet.get_action(observation, epsilon)
                eva_action = TrainNet.get_action_eva(observation, epsilon)

                if step > k:
                    eva_scores = TargetNet.eva(state_log, eva_index, k)
                    scores = [x + y for x, y in zip(eva_action, eva_scores)]
                    action = np.argmax(scores)

                action = action - 1
                prev_observation = observation

                observation = data.iloc[index + 1]
                price = observation[0]
                market_return = observation[2]

                new_portfolio_value = portfolio_value
                new_market_value = (price * buy_and_hold) + buy_and_hold_cash

                if stock_holdings >= trade_units and action == -1:
                    stock_holdings -= trade_units
                    cash += (price * trade_units)
                    new_portfolio_value = (stock_holdings * price) + cash
                elif cash >= (price*trade_units) and action == 1:
                    stock_holdings = stock_holdings + trade_units
                    cash -= (price * trade_units)
                    new_portfolio_value = (stock_holdings * price) + cash

                else:
                    action = 0
                    new_portfolio_value = (stock_holdings * price) + cash

                reward = math.log(new_portfolio_value/portfolio_value)
                rewards.append(reward)

                portfolio = np.array([stock_holdings, new_portfolio_value, new_market_value])
                observation = np.append(observation, portfolio)

                trade_log.loc[((episode * steps) + step)] = [price, action, stock_holdings, new_portfolio_value, new_market_value]

                portfolio_value = new_portfolio_value
                market_value = new_market_value

                if (step == steps-1) and (portfolio_value>market_value):
                    reward = (portfolio_value-market_value) * 100
                    print("End Price: ", price)
                elif (step == steps-1) and (portfolio_value<market_value):
                    reward = (portfolio_value-market_value) * 100
                    print("End Price: ", price)

                a = action + 1

                exp = {'s': prev_observation, 'a': a, 'r': reward, 's2': observation, 'done': done}
                TrainNet.add_experience(exp)
                TrainNet.train(TargetNet)

                if step % tau == 0:
                    TargetNet.copy_weights(TrainNet)

            advs.append(((portfolio_value / market_value) - 1))
            actions.append(action)

        if epsilon > .01:
            epsilon = epsilon - .009

        advantages.append(np.mean(advs))
        print("Sell: ", actions.count(-1), " Hold: ", actions.count(0), " Buy: ", (actions.count(1)-1))
        print("Cash: ", cash)
        print("Price: ", price)

        if episode % 1 == 0:
            print('Episode: {:5d} | Epsilon: {:1.3f} | '
                  'Agent Stocks : {:4d} | Total Agent Value: {:9.2f} | Buy and Hold Value: {:9.2f} | Agent Advantage %: {:+2.2f}'.format(
                episode, epsilon, stock_holdings, portfolio_value,  market_value, np.mean(advs)*100))

        print("------------------------------------------------------------------")


    test_cash = 100
    test_buy_hold_cash = 100
    test_holdings = 100
    test_buy_and_hold = 100
    test_portfolio_value = 1000
    test_market_value = 1000

    test_episode = []
    test_return = []
    agent_value = []
    market_value = []

    k = 10

    test_log = pd.DataFrame(index=np.arange(0, len(test_data.index)), columns=("Close", "Volume", "Returns", "Close_100",
                                                                        "Volume_100", "Close_20", "Volume_20",
                                                                        "Returns_5", "Returns_21", "RSI",
                                                                        "Holdings", "Portfolio_Val", "Market_Val"))

    for index in range(len(test_data.index)):
        test_episode.append(index)

        if index==0:
            observation = test_data.iloc[index]
            price = observation[0]
            test_portfolio_value = (price * test_holdings) + test_cash
            test_market_value = (price * test_buy_and_hold) + test_buy_hold_cash
            test_return.append(0)
            agent_value.append(test_portfolio_value)
            market_value.append(test_market_value)

        else:
            observation = test_data.iloc[index]
            portfolio = np.array([test_holdings, test_portfolio_value, test_market_value])
            observation = np.append(observation, portfolio)

            test_log.loc[index] = observation
            action = TrainNet.get_action(observation, epsilon)
            eva_action = TrainNet.get_action_eva(observation, epsilon)

            if index > k:
                eva_scores = TargetNet.eva(test_log, index, k)
                scores = [x + y for x, y in zip(eva_action, eva_scores)]
                action = np.argmax(scores)

            action = action - 1
            price = observation[0]
            test_market_return = observation[2]
            new_portfolio_value = test_portfolio_value
            new_market_value = (price * test_buy_and_hold) + test_buy_hold_cash

            if test_holdings >= trade_units and action == -1:
                test_holdings -= trade_units
                test_cash += (price * trade_units)
                new_portfolio_value = (test_holdings * price) + test_cash
            elif test_cash >= (price * trade_units) and action == 1:
                test_holdings = test_holdings + trade_units
                test_cash -= (price * trade_units)
                new_portfolio_value = (test_holdings * price) + test_cash

            else:
                action = 0
                new_portfolio_value = (test_holdings * price) + test_cash

            test_portfolio_value = new_portfolio_value
            test_market_value = new_market_value

            test_return.append((((test_portfolio_value/test_market_value)-1)*100))
            agent_value.append(test_portfolio_value)
            market_value.append(test_market_value)

        print("Test agent value: ", test_portfolio_value)
        print("Test stock holdings: ", test_holdings)
        print("Test market value: ", test_market_value)
        print("Comparison %: ", ((test_portfolio_value/test_market_value)-1)*100)
        print("------------------------------------------------------------------")


    plt.plot(test_episode, test_return)
    plt.xlabel('Trading Days')
    plt.ylabel('Agent Advantage %')
    plt.title(stock + " + EVA")
    plt.grid()
    plt.show()

    plt.plot(test_episode, agent_value)
    plt.xlabel('Trading Days')
    plt.ylabel('Agent Portfolio Value')
    plt.title(stock + " + EVA")
    plt.grid()
    plt.show()

    plt.plot(test_episode, agent_value, color='blue', label="Agent")
    plt.plot(test_episode, market_value, color='red', label="Buy and Hold")
    plt.xlabel('Trading Days')
    plt.ylabel('Portfolio Value')
    plt.title(stock + " + EVA")
    plt.grid()
    plt.legend()
    plt.show()

main()