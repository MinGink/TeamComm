#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Tushar Jain, Amanpreet Singh
Simulate a predator prey environment.
Each agent can just observe itself (it's own identity) i.e. s_j = j and vision sqaure around it.
Design Decisions:
    - Memory cheaper than time (compute)
    - Using Vocab for class of box:
         -1 out of bound,
         indexing for predator agent (from 2?)
         ??? for prey agent (1 for fixed case, for now)
    - Action Space & Observation Space are according to an agent
    - Rewards -0.05 at each time step till the time
    - Episode never ends
    - Obs. State: Vocab of 1-hot < predator, preys & units >
"""

# core modules
import random
import math

import curses

# 3rd party modules
import gym
import numpy as np
from gym import spaces


class PredatorPreyEnv(gym.Env):
    # metadata = {'render.modes': ['human']}

    def __init__(self,):
        self.__version__ = "0.0.1"

        # TODO: better config handling
        self.OUTSIDE_CLASS = 1
        self.PREY_CLASS = 2
        self.PREDATOR_CLASS = 3
        self.TIMESTEP_PENALTY = -0.05
        self.PREY_REWARD = 0.0
        self.POS_PREY_REWARD = 0.5
        self.Winner_REWARD = 1.0
        self.episode_over = False





    def init_curses(self):
        self.stdscr = curses.initscr()
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_RED, -1)
        curses.init_pair(2, curses.COLOR_YELLOW, -1)
        curses.init_pair(3, curses.COLOR_CYAN, -1)
        curses.init_pair(4, curses.COLOR_GREEN, -1)


    def init_args(self, parser):
        env = parser.add_argument_group('Prey Predator task')
        env.add_argument('--nenemies', type=int, default=2, help='Number of enemies')
        env.add_argument('--dim', type=int, default=5,
                         help="Dimension of box")
        env.add_argument('--vision', type=int, default=1,
                         help="Vision of predator")
        env.add_argument('--moving_prey', action="store_true", default=False,
                         help="Whether prey is fixed or moving")
        env.add_argument('--dynamic_prey', action="store_true", default=False,
                         help="Whether generate prey dynamically")
        parser.add_argument('--mode', default='old', type=str,
                        help='old|new')
        # env.add_argument('--enemy_comm', action="store_true", default=False,
        #                  help="Whether prey can communicate.")



    def multi_agent_init(self, args):

        # General variables defining the environment : CONFIG
        #params = ['dim', 'vision', 'moving_prey', 'mode', 'enemy_comm']
        params = ['dim', 'vision', 'moving_prey','mode',] #'enemy_comm']
        for key in params:
            setattr(self, key, getattr(args, key))

        self.nprey = args.nenemies
        self.npredator = args.nfriendly
        self.dims = dims = (self.dim, self.dim)
        self.moving_prey = args.moving_prey
        self.mode = args.mode

        # if args.moving_prey:
        #     raise NotImplementedError
        #     # TODO

        # (0: UP, 1: RIGHT, 2: DOWN, 3: LEFT, 4: STAY)
        # Define what an agent can do
        self.naction = 5

        self.action_space = spaces.MultiDiscrete([self.naction])
        self.prey_action_space = spaces.MultiDiscrete([self.naction] * self.nprey)

        if self.npredator // self.nprey >= 3:
            self.kill_num =self.npredator // self.nprey - 1
        else:
            self.kill_num = 2

        # if self.mode=='old':
        #     self.moving_prey = True


        self.BASE = (dims[0] * dims[1])
        self.OUTSIDE_CLASS += self.BASE
        self.PREY_CLASS += self.BASE
        self.PREDATOR_CLASS += self.BASE

        # Setting max vocab size for 1-hot encoding
        self.vocab_size = 1 + 1 + self.BASE + 1 + 1
        #          predator + prey + grid + outside

        # Observation for each agent will be vision * vision ndarray
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.vocab_size, (2 * self.vision) + 1, (2 * self.vision) + 1), dtype=int)
        # Actual observation will be of the shape 1 * npredator * (2v+1) * (2v+1) * vocab_size

        return


    def step(self, action):
        """
        The agents take a step in the environment.
        Parameters
        ----------
        action : list/ndarray of length m, containing the indexes of what lever each 'm' chosen agents pulled.
        Returns
        -------
        obs, reward, episode_over, info : tuple
            obs (object) :
            reward (float) : Ratio of Number of discrete levers pulled to total number of levers.
            episode_over (bool) : Will be true as episode length is 1
            info (dict) : diagnostic information useful for debugging.
        """
        if self.episode_over:
            raise RuntimeError("Episode is done")
        action = np.array(action).squeeze()
        action = np.atleast_1d(action)


        for i, a in enumerate(action):
            self._take_action(i, a)

        if self.moving_prey:
            prey_action = self.prey_action_space.sample()
            for i, a in enumerate(prey_action):
                self._prey_take_action(i, a)



        #assert np.all(action <= self.naction), "Actions should be in the range [0,naction)."

        # Update the state of the environment

        self.episode_over = False
        self.obs = self._get_obs()
        reward = self._get_reward()


        debug = {'predator_locs':self.predator_loc,'prey_locs':self.prey_loc}
        return self.obs, reward, self.episode_over, debug


    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.
        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.episode_over = False
        self.reached_prey = np.zeros(self.npredator)

        # Locations
        locs = self._get_cordinates()
        self.predator_loc, self.prey_loc = locs[:self.npredator], locs[self.npredator:]

        self._set_grid()

        # stat - like success ratio
        self.stat = dict()

        # Observation will be npredator * vision * vision ndarray
        self.obs = self._get_obs()
        return self.obs

    def seed(self):
        return


    def _get_cordinates(self):
        idx = np.random.choice(np.prod(self.dims),(self.npredator + self.nprey), replace=False)
        return np.vstack(np.unravel_index(idx, self.dims)).T


    def _set_grid(self):
        self.grid = np.arange(self.BASE).reshape(self.dims)
        # Mark agents in grid
        # self.grid[self.predator_loc[:,0], self.predator_loc[:,1]] = self.predator_ids
        # self.grid[self.prey_loc[:,0], self.prey_loc[:,1]] = self.prey_ids

        # Padding for vision
        self.grid = np.pad(self.grid, self.vision, 'constant', constant_values = self.OUTSIDE_CLASS)

        self.empty_bool_base_grid = self._onehot_initialization(self.grid)


    def _get_obs(self):
        self.bool_base_grid = self.empty_bool_base_grid.copy()

        for i, p in enumerate(self.predator_loc):
            self.bool_base_grid[p[0] + self.vision, p[1] + self.vision, self.PREDATOR_CLASS] += 1

        for i, p in enumerate(self.prey_loc):
            self.bool_base_grid[p[0] + self.vision, p[1] + self.vision, self.PREY_CLASS] += 1

        obs = []
        for p in self.predator_loc:
            slice_y = slice(p[0], p[0] + (2 * self.vision) + 1)
            slice_x = slice(p[1], p[1] + (2 * self.vision) + 1)
            obs.append(self.bool_base_grid[slice_y, slice_x])

        # if self.enemy_comm:
        #     for p in self.prey_loc:
        #         slice_y = slice(p[0], p[0] + (2 * self.vision) + 1)
        #         slice_x = slice(p[1], p[1] + (2 * self.vision) + 1)
        #         obs.append(self.bool_base_grid[slice_y, slice_x])

        obs = np.stack(obs)
        return obs



    def _prey_take_action(self, idx, act):
        # STAY action
        if act==4:
            return

        # UP
        if act==0 and self.grid[max(0,
                                self.prey_loc[idx][0] + self.vision - 1),
                                self.prey_loc[idx][1] + self.vision] != self.OUTSIDE_CLASS:
            self.prey_loc[idx][0] = max(0, self.prey_loc[idx][0]-1)

        # RIGHT
        elif act==1 and self.grid[self.prey_loc[idx][0] + self.vision,
                                min(self.dims[1] -1,
                                    self.prey_loc[idx][1] + self.vision + 1)] != self.OUTSIDE_CLASS:
            self.prey_loc[idx][1] = min(self.dims[1]-1,
                                            self.prey_loc[idx][1]+1)

        # DOWN
        elif act==2 and self.grid[min(self.dims[0]-1,
                                    self.prey_loc[idx][0] + self.vision + 1),
                                    self.prey_loc[idx][1] + self.vision] != self.OUTSIDE_CLASS:
            self.prey_loc[idx][0] = min(self.dims[0]-1,
                                            self.prey_loc[idx][0]+1)

        # LEFT
        elif act==3 and self.grid[self.prey_loc[idx][0] + self.vision,
                                    max(0,
                                    self.prey_loc[idx][1] + self.vision - 1)] != self.OUTSIDE_CLASS:
            self.prey_loc[idx][1] = max(0, self.prey_loc[idx][1]-1)


    def _take_action(self, idx, act):

        # STAY action
        if act == 4:
            return

        # UP
        if act == 0 and self.grid[max(0,
                                      self.predator_loc[idx][0] + self.vision - 1),
                                  self.predator_loc[idx][1] + self.vision] != self.OUTSIDE_CLASS:
            self.predator_loc[idx][0] = max(0, self.predator_loc[idx][0] - 1)

        # RIGHT
        elif act == 1 and self.grid[self.predator_loc[idx][0] + self.vision,
                                    min(self.dims[1] - 1,
                                        self.predator_loc[idx][1] + self.vision + 1)] != self.OUTSIDE_CLASS:
            self.predator_loc[idx][1] = min(self.dims[1] - 1,
                                            self.predator_loc[idx][1] + 1)

        # DOWN
        elif act == 2 and self.grid[min(self.dims[0] - 1,
                                        self.predator_loc[idx][0] + self.vision + 1),
                                    self.predator_loc[idx][1] + self.vision] != self.OUTSIDE_CLASS:
            self.predator_loc[idx][0] = min(self.dims[0] - 1,
                                            self.predator_loc[idx][0] + 1)

        # LEFT
        elif act == 3 and self.grid[self.predator_loc[idx][0] + self.vision,
                                    max(0,
                                        self.predator_loc[idx][1] + self.vision - 1)] != self.OUTSIDE_CLASS:
            self.predator_loc[idx][1] = max(0, self.predator_loc[idx][1] - 1)


    # def _get_reward(self):
    #     n = self.npredator if not self.enemy_comm else self.npredator + self.nprey
    #     reward = np.full(n, self.TIMESTEP_PENALTY)
    #
    #     on_prey = np.where(np.all(self.predator_loc == self.prey_loc,axis=1))[0]
    #     nb_predator_on_prey = on_prey.size
    #
    #     if self.mode == 'cooperative':
    #         reward[on_prey] = self.POS_PREY_REWARD * nb_predator_on_prey
    #     elif self.mode == 'competitive':
    #         if nb_predator_on_prey:
    #             reward[on_prey] = self.POS_PREY_REWARD / nb_predator_on_prey
    #     elif self.mode == 'mixed':
    #         reward[on_prey] = self.PREY_REWARD
    #     else:
    #         raise RuntimeError("Incorrect mode, Available modes: [cooperative|competitive|mixed]")
    #
    #     self.reached_prey[on_prey] = 1
    #
    #     if np.all(self.reached_prey == 1) and self.mode == 'mixed':
    #         self.episode_over = True
    #
    #     # Prey reward
    #     if nb_predator_on_prey == 0:
    #         reward[self.npredator:] = -1 * self.TIMESTEP_PENALTY
    #     else:
    #         # TODO: discuss & finalise
    #         reward[self.npredator:] = 0
    #
    #     # Success ratio
    #     if self.mode != 'competitive':
    #         if nb_predator_on_prey == self.npredator:
    #             self.stat['success'] = 1
    #         else:
    #             self.stat['success'] = 0
    #
    #     return reward


    def _get_reward(self):
        self.stat['success'] = 0

        n = self.npredator  # if not self.enemy_comm else self.npredator + self.nprey
        reward = np.full(n, self.TIMESTEP_PENALTY)


        if self.mode == 'old':
            self.reached_prey = np.zeros(self.npredator)
            for i in range(self.prey_loc.shape[0]):
                on_prey_tmp = np.where(np.all(self.predator_loc == self.prey_loc[i], axis=1))[0]
                reward[on_prey_tmp] = self.PREY_REWARD
                self.reached_prey[on_prey_tmp] = 1
            if np.all(self.reached_prey == 1):
                self.episode_over = True


        elif self.mode == 'new':
            dead_pery = []
            for i in range(self.prey_loc.shape[0]):
                on_prey_tmp = np.where(np.all(self.predator_loc == self.prey_loc[i], axis=1))[0]
                reward[on_prey_tmp] = self.PREY_REWARD
                if on_prey_tmp.size >= self.kill_num :
                    reward[on_prey_tmp] = self.POS_PREY_REWARD / on_prey_tmp.size
                    dead_pery.append(i)
            self.prey_loc = np.delete(self.prey_loc, dead_pery, axis=0)
            if self.prey_loc.size == 0:
                self.episode_over = True
                self.stat['success'] = 1
        return reward


    def _prey_reset(self, idx):
        prey_idx = np.random.choice(np.prod(self.dims), (1), replace=False)
        self.prey_loc[idx] = np.vstack(np.unravel_index(prey_idx, self.dims)).T
        #self._set_grid()


    def reward_terminal(self):
        return np.zeros_like(self._get_reward())


    def _onehot_initialization(self, a):
        ncols = self.vocab_size
        out = np.zeros(a.shape + (ncols,), dtype=int)
        out[self._all_idx(a, axis=2)] = 1
        return out


    def _all_idx(self, idx, axis):
        grid = np.ogrid[tuple(map(slice, idx.shape))]
        grid.insert(axis, idx)
        return tuple(grid)





    def render(self, mode='human', close=False):
        grid = np.zeros(self.BASE, dtype=object).reshape(self.dims)
        self.stdscr.clear()

        for p in self.predator_loc:
            if grid[p[0]][p[1]] != 0:
                grid[p[0]][p[1]] = str(grid[p[0]][p[1]]) + 'X'
            else:
                grid[p[0]][p[1]] = 'X'

        for p in self.prey_loc:
            if grid[p[0]][p[1]] != 0:
                grid[p[0]][p[1]] = str(grid[p[0]][p[1]]) + 'P'
            else:
                grid[p[0]][p[1]] = 'P'

        for row_num, row in enumerate(grid):
            for idx, item in enumerate(row):
                if item != 0:
                    if 'X' in item and 'P' in item:
                        self.stdscr.addstr(row_num, idx * 4, item.center(3), curses.color_pair(3))
                    elif 'X' in item:
                        self.stdscr.addstr(row_num, idx * 4, item.center(3), curses.color_pair(1))
                    else:
                        self.stdscr.addstr(row_num, idx * 4, item.center(3),  curses.color_pair(2))
                else:
                    self.stdscr.addstr(row_num, idx * 4, '0'.center(3), curses.color_pair(4))

        self.stdscr.addstr(len(grid), 0, '\n')
        self.stdscr.refresh()

    def exit_render(self):
        curses.endwin()