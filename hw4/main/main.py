import random
from collections import defaultdict, namedtuple
from enum import Enum
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np
from itertools import product


class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


GridState = namedtuple('GridState', ['x', 'y', 'cows_vec'])


class MDP:
    """
            (9, 9)
         ...
    (0,0)
    """
    width = 10
    height = 10

    @staticmethod
    def _try_move(s, a):
        return {
            Action.UP: GridState(s.x, s.y + 1, s.cows_vec),
            Action.DOWN: GridState(s.x, s.y - 1, s.cows_vec),
            Action.LEFT: GridState(s.x - 1, s.y, s.cows_vec),
            Action.RIGHT: GridState(s.x + 1, s.y, s.cows_vec)
        }[a]

    def _move(self, cell, direction):
        next_cell = self._try_move(cell, direction)
        if next_cell.x not in range(0, self.width) or next_cell.y not in range(0, self.height):
            return cell
        if (next_cell.x, next_cell.y) in self.cow_positions:
            cow_ind = self.cow_positions.index((next_cell.x, next_cell.y))
            new_cow_vec = tuple(True if i == cow_ind else e for i, e in enumerate(cell.cows_vec))
            next_cell = GridState(next_cell.x, next_cell.y, new_cow_vec)

        return next_cell

    def _get_actions(self):
        actions = {}
        for s in self.states:
            actions[s] = []
            if s.x > 0:
                actions[s].append(Action.LEFT)
            if s.x < self.width - 1:
                actions[s].append(Action.RIGHT)
            if s.y > 0:
                actions[s].append(Action.DOWN)
            if s.y < self.width - 1:
                actions[s].append(Action.UP)
        return actions

    @staticmethod
    def reverse_action(action):
        return {Action.UP: Action.DOWN,
                Action.DOWN: Action.UP,
                Action.LEFT: Action.RIGHT,
                Action.RIGHT: Action.LEFT
                }[action]

    def _get_reward(self):
        reward_partial = {}

        for s in self.states:
            for a in self.actions[s]:
                next_state = self.transitions[(s, a)]
                if next_state == self.END_STATE:
                    reward_partial[(s, a)] = 100
                if (next_state.x, next_state.y) in self.cow_positions:
                    cow_ind = self.cow_positions.index((next_state.x, next_state.y))
                    if not s.cows_vec[cow_ind]:
                        reward_partial[(s, a)] = 50

        return defaultdict(int, reward_partial)

    def _get_states(self):
        states = []
        for x in range(self.width):
            for y in range(self.height):
                # cow_vectors = product([False, True], repeat=len(self.cow_positions))
                ncows = len(self.cow_positions)
                if (x, y) not in self.cow_positions:
                    cow_vectors = tuple(product([False, True], repeat=ncows))
                else:
                    cow_ind = self.cow_positions.index((x, y))
                    cow_vectors = tuple(prod for prod in product([False, True], repeat=ncows) if prod[cow_ind])
                for cw in cow_vectors:
                    states.append(GridState(x, y, cw))

        return states

    def __init__(self, lion_pos=(0, 0), cow_positions=([(9, 9)])):
        self.lion_x, self.lion_y = lion_pos
        self.cow_positions = cow_positions

        self.START_STATE = GridState(self.lion_x, self.lion_y, tuple(False for _ in cow_positions))
        self.END_STATE = GridState(self.lion_x, self.lion_y, tuple(True for _ in cow_positions))
        self.states = self._get_states()
        self.actions = self._get_actions()
        self.transitions = {(s, a): self._move(s, a) for s in self.states for a in self.actions[s]}
        self.reward = self._get_reward()


def get_policy(mdp, quality, eps=0.3):
    policy = {}
    for s in mdp.states:
        best_act_ind = np.argmax([quality[(s, a)] for a in mdp.actions[s]])
        best_act = mdp.actions[s][best_act_ind]
        for a in mdp.actions[s]:
            policy[(s, a)] = 1 - eps + eps / len(mdp.actions[s]) if a == best_act \
                else eps / len(mdp.actions[s])
    return policy


def get_action(mdp, policy, s):
    prob = random.uniform(0, 1)
    for a in mdp.actions[s]:
        if prob <= policy[(s, a)]:
            return a
        prob -= policy[(s, a)]
    raise ValueError("Policy probabilities should sum up to 1")


def sarsa(mdp, n_episodes, alpha, gamma):
    timestampes = []
    nepisodes = []

    quality = {}
    for s in mdp.states:
        for a in mdp.actions[s]:
            quality[(s, a)] = 1

    time_cnt = 0
    epi_cnt = 0
    for _ in range(n_episodes):
        s = mdp.START_STATE
        a = random.choice(mdp.actions[s])
        while s != mdp.END_STATE:
            r = mdp.reward[(s, a)]
            ss = mdp.transitions[(s, a)]
            policy = get_policy(mdp, quality)
            aa = get_action(mdp, policy, ss)
            quality[(s, a)] += alpha * (r + gamma * quality[(ss, aa)] - quality[(s, a)])
            s = ss
            a = aa
            timestampes.append(time_cnt)
            nepisodes.append(epi_cnt)

            time_cnt += 1
        epi_cnt += 1
    return get_policy(mdp, quality), timestampes, nepisodes


def generate_episode(mdp, policy):
    episode = []
    reward = 0
    rewards = []

    s = mdp.START_STATE
    a = random.choice(mdp.actions[s])

    while s != mdp.END_STATE:
        episode.append((s, a))
        r = mdp.reward[(s, a)]
        rewards.append(r)
        ss = mdp.transitions[(s, a)]
        aa = get_action(mdp, policy, ss)
        s = ss
        a = aa
        reward += r

    rewards_cumsum = np.cumsum(rewards[::-1])[::-1]
    reward_after = {}
    for (s, a), r in reversed(list(zip(episode, rewards_cumsum))):
        if (s, a) not in reward_after:
            reward_after[(s, a)] = r

    return episode, reward, defaultdict(int, reward_after)


def on_policy_first_visit_mc_control(mdp, eps, n_episodes):
    timestampes = []
    nepisodes = []

    quality = {}
    returns = {}
    policy = {}
    for s in mdp.states:
        for a in mdp.actions[s]:
            quality[(s, a)] = -2
            returns[(s, a)] = []
            policy[(s, a)] = 1 / len(mdp.actions[s])

    time_cnt = 0
    epi_cnt = 0
    while True:
        episode, reward, reward_after = generate_episode(mdp, policy)

        for _ in episode:
            nepisodes.append(epi_cnt)
            timestampes.append(time_cnt)
            time_cnt += 1
        epi_cnt += 1

        episode_unique = list(set(episode))
        for s, a in episode_unique:
            returns[(s, a)].append(reward_after[(s, a)])
            quality[(s, a)] = np.mean(returns[(s, a)])
        epi_states = [s for s, _ in episode_unique]

        for s in epi_states:
            best_act_ind = np.argmax([quality[(s, a)] for a in mdp.actions[s]])
            best_act = mdp.actions[s][best_act_ind]
            for a in mdp.actions[s]:
                policy[(s, a)] = 1 - eps + eps / len(mdp.actions[s]) if a == best_act \
                    else eps / len(mdp.actions[s])

        epi_cnt += 1
        if epi_cnt > n_episodes:
            return policy, timestampes, nepisodes


def value_iteration(mdp, gamma, theta):
    values = {}
    for s in mdp.states:
        values[s] = 1
    niter = 0
    while True:
        niter += 1
        delta = 0
        for state in mdp.states:
            v = values[state]
            candidates = [r + gamma * values[ss]
                          for a in mdp.actions[state]
                          for ss in [mdp.transitions[(state, a)]] for r in [mdp.reward[(state, a)]]]
            values[state] = max(candidates)
            delta = max(delta, abs(v - values[state]))
        if delta < theta:
            print("Number of iterations", niter)
            return values


def plot_policy(policy_vals):
    skeys = sorted(policy_vals.keys())
    x = np.array([a for a, _ in skeys])
    y = np.array([b for _, b in skeys])

    temp = np.array([policy_vals[k] for k in skeys])

    nrows, ncols = 10, 10
    grid = temp.reshape((nrows, ncols))

    plt.imshow(grid, extent=(x.min(), x.max(), y.max(), y.min()), cmap=cm.Greys)
    plt.show()


def main():
    pass


if __name__ == '__main__':
    main()
