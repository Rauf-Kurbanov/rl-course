from collections import defaultdict, namedtuple
from enum import Enum
from functools import total_ordering

from itertools import product
import random


@total_ordering
class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented


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
                next_state = self._transitions_dict[(s, a)]
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
                ncows = len(self.cow_positions)
                if (x, y) not in self.cow_positions:
                    cow_vectors = tuple(product([False, True], repeat=ncows))
                else:
                    cow_ind = self.cow_positions.index((x, y))
                    cow_vectors = tuple(prod for prod in product([False, True], repeat=ncows) if prod[cow_ind])
                for cw in cow_vectors:
                    states.append(GridState(x, y, cw))

        return states

    def reward_transition(self, s, a):
        ss = self._transitions_dict[(s, a)]
        r = self._reward_dict[(s, a)]
        return r, ss

    def __init__(self, lion_pos=(0, 0), cow_positions=([(9, 9)])):
        self.lion_x, self.lion_y = lion_pos
        self.cow_positions = cow_positions

        self.START_STATE = GridState(self.lion_x, self.lion_y, tuple(False for _ in cow_positions))
        self.END_STATE = GridState(self.lion_x, self.lion_y, tuple(True for _ in cow_positions))
        self.states = self._get_states()
        self.actions = self._get_actions()

        self._transitions_dict = {(s, a): self._move(s, a) for s in self.states for a in self.actions[s]}
        self._reward_dict = self._get_reward()


class StochasticMDP(MDP):

    @staticmethod
    def _opposite(action):
        return {Action.UP: Action.DOWN,
                Action.DOWN: Action.UP,
                Action.LEFT: Action.RIGHT,
                Action.RIGHT: Action.LEFT}[action]

    def _get_transitions(self, success_prob):
        def transitions_func(s, a):
            prob = random.uniform(0, 1)
            if prob <= success_prob:
                actual_action = a
            else:
                actual_action = self._opposite(a)
            return self._move(s, actual_action), actual_action

        return transitions_func

    def reward_transition(self, s, a):
        ss, actual_action = self._transitions_prob(s, a)
        r = self._reward_dict[(s, actual_action)]
        return r, ss

    def __init__(self, lion_pos=(0, 0), cow_positions=([(9, 9)]), success_prob=0.7):
        super().__init__(lion_pos, cow_positions)

        self._transitions_prob = self._get_transitions(success_prob)
