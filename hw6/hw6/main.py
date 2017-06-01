import random
import gym
import numpy as np

from hw6.tile_manager import TileManader


def qval(tm, observation, action, weights):
    return np.dot(tm.features(observation, action).reshape(1, tm.maxtiles), weights)[0]


def epsilon_greedy(env, observation, epsilon, tm, weights):
    if random.random() > epsilon:
        qval_list = []
        for act in range(env.action_space.n):
            qval_list.append(qval(tm, observation, act, weights))
        return np.argmax(qval_list)
    else:
        return random.randint(0, 2)


def semi_gradient_sarsa(env, tm, alpha, gamma, epsilon, maxtiles, maxiter=5000, show_each=10):
    theta_weights = np.zeros(maxtiles)

    for i_episode in range(maxiter):
        observation = env.reset()
        reward_total = 0
        action = epsilon_greedy(env, observation, epsilon, tm, theta_weights)
        while True:
            observation_n, reward_n, done, info = env.step(action)
            if i_episode % show_each == 0:
                env.render()
            reward_total += reward_n
            if done:
                theta_weights += (alpha * (reward_n - qval(tm, observation, action, theta_weights))) \
                                 * tm.delta(observation, action)
                break
            action_n = epsilon_greedy(env, observation_n, epsilon, tm, theta_weights)
            q_n = qval(tm, observation_n, action_n, theta_weights)
            q = qval(tm, observation, action, theta_weights)
            theta_weights += (alpha * (reward_n + gamma * q_n - q)) * tm.delta(observation, action)
            observation = observation_n
            action = action_n


def main():
    env = gym.make('MountainCar-v0')

    maxtiles = 2048
    tm = TileManader(env, maxtiles=maxtiles)
    semi_gradient_sarsa(env, tm, alpha=0.01, gamma=1., epsilon=0.2, maxtiles=maxtiles)


if __name__ == '__main__':
    main()
