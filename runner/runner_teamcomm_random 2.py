import torch
import torch.nn as nn
from collections import namedtuple
import numpy as np
from torch.optim import Adam,RMSprop
from modules.utils import merge_dict, multinomials_log_density
import time
import random
from runner import Runner

import argparse

Transition = namedtuple('Transition', ('action_outs', 'actions', 'rewards', 'values', 'episode_masks', 'episode_agent_masks'))


class RunnerTeamcommRandom(Runner):
    def __init__(self, config, env, agent):
        super().__init__(config, env, agent)


        self.optimizer_agent_ac = RMSprop(self.agent.agent.parameters(), lr = self.args.lr, alpha=0.97, eps=1e-6)

        self.n_nodes = int(self.n_agents * (self.n_agents - 1) / 2)
        self.interval = self.args.interval


    def optimizer_zero_grad(self):
        self.optimizer_agent_ac.zero_grad()


    def optimizer_step(self):
        self.optimizer_agent_ac.step()



    def train_batch(self, batch_size):
        batch_data, batch_log = self.collect_batch_data(batch_size)
        self.optimizer_zero_grad()
        train_log = self.compute_grad(batch_data)
        merge_dict(batch_log, train_log)
        for p in self.params:
            if p._grad is not None:
                p._grad.data /= batch_log['num_steps']
        self.optimizer_step()
        return train_log




    def run_an_episode(self):

        log = dict()

        memory = []

        self.reset()
        obs = self.env.get_obs()

        obs_tensor = torch.tensor(np.array(obs), dtype=torch.float)
        team_action_out = self.agent.teaming(obs_tensor)
        team_action = self.choose_action(team_action_out)

        step = 1
        num_group = 0
        episode_return = 0
        done = False

        while not done and step <= self.args.episode_length:

            obs_tensor = torch.tensor(np.array(obs), dtype=torch.float)

            if step % self.interval == 0:
                team_action_out = self.agent.teaming(obs_tensor)
                team_action = self.choose_action(team_action_out)


            sets = self.matrix_to_set(team_action)
            after_comm, mu, std = self.agent.communicate(obs_tensor, sets)
            action_outs, values = self.agent.agent(after_comm)
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
            num_group += len(sets)

        log['episode_return'] = episode_return
        log['episode_steps'] = [step - 1]
        log['num_groups'] = num_group / (step - 1)

        if 'num_collisions' in env_info:
            log['num_collisions'] = int(env_info['num_collisions'])

        return memory, log


def random_sets(n):
    agent_list = list(range(n))
    #random divide the agents in the list into different groups
    random.shuffle(agent_list)
    sets = []
    while len(agent_list) > 0:
        num = random.randint(1, len(agent_list))
        sets.append(agent_list[:num])
        agent_list = agent_list[num:]
    return sets