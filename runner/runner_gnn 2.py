import torch
import torch.nn as nn
from collections import namedtuple
import numpy as np
from torch.optim import Adam,RMSprop
from modules.utils import merge_dict, multinomials_log_density
import time
from .runner import Runner

import argparse

Transition = namedtuple('Transition', ('action_outs', 'actions', 'rewards', 'values', 'episode_masks', 'episode_agent_masks'))


class RunnerGNN(Runner):
    def __init__(self, config, env, agent):
        super().__init__(config, env, agent)



    def run_an_episode(self):

        memory = []
        log = dict()
        episode_return = 0

        self.reset()
        obs = self.env.get_obs()

        step = 1
        done = False
        while not done and step <= self.args.episode_length:

            obs_tensor = torch.tensor(np.array(obs), dtype=torch.float)
            graph = self.env.get_graph()
            action_outs, values = self.agent(obs_tensor, graph)
            actions = self.choose_action(action_outs)
            rewards, done, env_info = self.env.step(actions)
            next_obs = self.env.get_obs()

            done = done or step == self.args.episode_length

            episode_mask = np.ones(rewards.shape)
            episode_agent_mask = np.ones(rewards.shape)
            if done:
                episode_mask = np.zeros(rewards.shape)
            elif 'completed_agent' in env_info:
                episode_agent_mask = 1 - np.array(env_info['completed_agent']).reshape(-1)


            trans = Transition(action_outs, actions, rewards, values, episode_mask, episode_agent_mask)
            memory.append(trans)

            obs = next_obs
            episode_return += int(np.sum(rewards))
            step += 1


        log['episode_return'] = episode_return
        log['episode_steps'] = [step-1]

        if 'num_collisions' in env_info:
            log['num_collisions'] = int(env_info['num_collisions'])

        return memory, log


