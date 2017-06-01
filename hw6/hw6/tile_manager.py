import numpy as np

from utils.TileCoding import IHT, tiles


class TileManader():
    def __init__(self, env, maxtiles=2048, numtilings=8):
        self.maxtiles = maxtiles
        self.numtilings = numtilings

        max_position, max_velocity = env.observation_space.high
        min_position, min_velocity = env.observation_space.low

        position_scale = numtilings / (max_position - min_position)
        velocity_scale = numtilings / (max_velocity - min_velocity)

        self.position_scale = position_scale
        self.velocity_scale = velocity_scale
        self.hash_table = IHT(maxtiles)

    def get_active_tiles(self, position, velocity, action):
        active_tiles = tiles(self.hash_table, self.numtilings,
                             [self.position_scale * position, self.velocity_scale * velocity],
                             [action])
        return active_tiles

    def features(self, observation, action):
        active_tiles = self.get_active_tiles(observation[0], observation[1], action)
        feature = np.zeros(self.maxtiles)
        for idx in active_tiles:
            feature[idx] = 1
        return feature

    def delta(self, observation, action):
        return self.features(observation, action)
