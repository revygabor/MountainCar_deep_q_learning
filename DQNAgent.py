from collections import deque
import random
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.models import save_model, load_model
from keras.optimizers import Adam


class DQNAgent:
    def __init__(self, state_size, action_size, load_model=False, train = False):
        self.state_size = state_size
        self.action_size = action_size
        self.load_model = load_model
        self.memory = deque([], 2096)
        self.gamma = 0.9            # Discount rate
        if train:
            self.epsilon = 1.0      # Explore rate
        else:
            self.epsilon = 0

        self.epsilon_decay = 0.998
        self.epsilon_min = 0.05
        self.learning_rate = 0.0001
        self.model = self.build_model()
        self.model_target = self.build_model()
        self.train_episode = 0      # At specific intervals we update the target network


    def build_model(self):
        if self.load_model:
            return load_model('model.h5')

        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))

        model.compile(loss='mse', optimizer='sgd')#Adam(lr=self.learning_rate))

        return model


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])


    def predict(self, x, target=False):
        if target:
            return self.model_target.predict(x)

        return self.model.predict(x)


    def train(self, batch_size):
        if self.train_episode % 10 is 0:
            self.updateTargetModel()

        minibatch = random.sample(self.memory, min(batch_size, len(self.memory)))

        for state, action, reward, next_state, done in minibatch:
            if done:
                target = reward

            else:
                target = reward + self.gamma * np.amax(self.predict(next_state, target=True)[0])

            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.train_episode += 1


    def updateTargetModel(self):
        self.model_target.set_weights(self.model.get_weights())


    def save_model(self):
        save_model(self.model, 'model.h5')
