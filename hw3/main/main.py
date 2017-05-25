from collections import namedtuple
from enum import Enum
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns


class Rank(Enum):
    A = 1
    R2 = 2
    R3 = 3
    R4 = 4
    R5 = 5
    R6 = 6
    R7 = 7
    R8 = 8
    R9 = 9
    R10 = 10
    J = 11
    Q = 12
    K = 13


def Card(rank):
    CT = namedtuple('CT', ['value', 'is_ace'])
    if rank == Rank.A:
        return CT(1, True)
    if rank in [Rank.J, Rank.Q, Rank.K]:
        return CT(10, False)
    return CT(rank.value, False)


class Action(Enum):
    HIT = 0
    STICK = 1


class BlackjackGame:
    def __init__(self):
        self.cards = [Card(rank) for rank in Rank]

        self.BjState = namedtuple('BjState', ['curr_sum', 'dealer_card', 'usable_ace'])
        self.states = [self.BjState(curr_sum, dealer_card, usable_ace)
                       for dealer_card in self.cards
                       for curr_sum in range(12, 22)
                       for usable_ace in ([True, False] if curr_sum <= 12 else [False])]

    @staticmethod
    def actions(state):
        return [Action.HIT, Action.STICK] if state.curr_sum < 21 else [Action.STICK]


def is_terminal(state):
    if state.usable_ace and state.curr_sum == 11:
        return True
    return state.curr_sum >= 21


def play_by_policy(mc, s, policy, episode):
    while not is_terminal(s):
        stick_prob = random.uniform(0, 1)
        if stick_prob <= policy[(s, Action.STICK)]:
            episode.append((s, Action.STICK))
            break

        episode.append((s, Action.HIT))
        next_card = random.choice(mc.cards)

        new_sum = s.curr_sum + next_card.value
        if_usable_ace = new_sum <= 11 and (next_card.is_ace or s.usable_ace)
        s = mc.BjState(new_sum, s.dealer_card, if_usable_ace)

    if s.curr_sum == 21:
        episode.append((s, Action.STICK))

    return s


def generate_episode(mc, policy, start_s):
    episode = []
    s = start_s

    s = play_by_policy(mc, s, policy, episode)

    # dealer's turn
    next_card = random.choice(mc.cards)
    dealer_sum = s.dealer_card.value + next_card.value
    dealer_ace_sum = dealer_sum + 10

    dealer_win_using_ace = (next_card.is_ace or s.dealer_card.is_ace) \
                           and s.curr_sum <= dealer_ace_sum <= 21
    lose_conditions = [dealer_win_using_ace,
                       s.curr_sum <= dealer_sum <= 21,
                       s.curr_sum >= 21]
    if any(lose_conditions):
        return episode, -1
    return episode, +1


def on_policy_first_visit_mc_control(mc, eps, max_iter):
    quality = {}
    returns = {}
    policy = {}
    for s in mc.states:
        for a in mc.actions(s):
            quality[(s, a)] = 0
            returns[(s, a)] = []
            policy[(s, a)] = 1 / len(mc.actions(s))

    niter = 0
    while True:
        start_state = random.choice(mc.states)
        episode, reward = generate_episode(mc, policy, start_state)

        for s, a in episode:
            returns[(s, a)].append(reward)
            quality[(s, a)] = np.mean(returns[(s, a)])

        epi_states = [s for s, _ in episode]
        epi_states = list(set(epi_states))
        for s in epi_states:
            best_act_ind = np.argmax([quality[(s, a)] for a in mc.actions(s)])
            best_act = mc.actions(s)[best_act_ind]
            for a in mc.actions(s):
                policy[(s, a)] = 1 - eps + eps / len(mc.actions(s)) if a == best_act \
                    else eps / len(mc.actions(s))

        niter += 1
        if niter > max_iter:
            return policy


def win_rate(mc, policy, n_trials=1000):
    n_wins = 0
    for _ in range(n_trials):
        start_state = random.choice(mc.states)
        episode, reward = generate_episode(mc, policy, start_state)
        if reward == 1:
            n_wins += 1
    return n_wins / n_trials


def main():
    random.seed(42)
    mc = BlackjackGame()
    iters = np.logspace(0, 4)
    win_rates = [rate for n_iter in iters
                 for policy in [on_policy_first_visit_mc_control(mc, eps=0.4, max_iter=n_iter)]
                 for rate in [win_rate(mc, policy)]]

    sns.set_style("darkgrid")
    plt.plot(iters, win_rates)
    plt.xlabel('Num of iterations')
    plt.ylabel('Win rate')
    plt.title('On policy first visit MC blackjack')
    plt.show()

if __name__ == '__main__':
    main()
