import torch
import torch.nn as nn
from collections import namedtuple
import numpy as np
from torch.optim import Adam
from inspect import getargspec
from modules.utils import merge_dict
from .runner import Runner
import time

Transition = namedtuple('Transition', ('action_outs', 'actions', 'rewards', 'values', 'episode_masks', 'episode_agent_masks'))
class RunnerMagic(Runner):
    def __init__(self, args, env, agent):
        super(RunnerMagic, self).__init__(args, env, agent)

        # self.args = args
        #
        # self.env = env
        # # self.agent = agent_REGISTRY[args.algo](args)
        # self.agent = agent
        # self.total_steps = 0
        #
        # self.gamma = self.args.gamma
        # self.lamda = self.args.lamda


        # self.transition = namedtuple('Transition', ('state', 'obs',
        #                                             'actions','action_outs','rewards',
        #                                             'episode_masks', 'episode_agent_masks','values'))
        #
        #
        #
        # self.params = list(self.agent.parameters())
        # self.optimizer = Adam(params=self.params, lr=args.lr)



    def run_an_episode(self):
        memory = []
        info = dict()
        log = dict()
        # episode_return = np.zeros(self.n_agents)
        episode_return = 0

        self.reset()

        obs = self.env.get_obs()

        prev_hid = torch.zeros(1, self.args.n_agents, self.args.hid_size)
        # prev_hid = self.agent.init_hidden(batch_size=state.shape[0])
        step = 1
        done = False
        while not done and step <= self.args.episode_length:
            misc = dict()
            if step == 1:
                prev_hid = self.agent.init_hidden(batch_size=1)
                # prev_hid = self.agent.init_hidden(batch_size=1)

        # for t in range(self.args.episode_length):
        #     if t == 0:

            obs_tensor = torch.tensor(np.array(obs), dtype=torch.float)
            obs_tensor = obs_tensor.unsqueeze(0)#
            obs_tensor = [obs_tensor, prev_hid]#
            action_outs, values, prev_hid = self.agent(obs_tensor, info)

            if step % self.args.detach_gap == 0:
                prev_hid = (prev_hid[0].detach(), prev_hid[1].detach())

            actions = self.choose_action(action_outs)

            rewards, done, env_info = self.env.step(actions)

            # next_state = self.env.get_state()
            next_obs = self.env.get_obs()
            done = done or step == self.args.episode_length

            episode_mask = np.ones(rewards.shape)
            episode_agent_mask = np.ones(rewards.shape)

            if done:
                episode_mask = np.zeros(rewards.shape)
            else:
                if 'is_completed' in env_info:
                    episode_agent_mask = 1 - env_info['is_completed'].reshape(-1)

            trans = Transition(action_outs, actions, rewards, values, episode_mask, episode_agent_mask)
            memory.append(trans)


            obs = next_obs

            # episode_return += rewards.astype(episode_return.dtype)
            episode_return += int(np.sum(rewards))
            step += 1

            # episode_return += float(sum(rewards))
            # self.total_steps += 1
        log['episode_return'] = episode_return
        log['episode_steps'] = [step - 1]
        if 'num_collisions' in env_info:
            log['num_collisions'] = int(env_info['num_collisions'])

        # if self.args.env == 'tj':
        #     merge_dict(self.env.get_stat(), log)
        return memory, log


