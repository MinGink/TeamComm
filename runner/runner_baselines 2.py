import torch
import torch.nn as nn
from collections import namedtuple
import numpy as np
from torch.optim import Adam
from modules.utils import merge_dict
import time
import argparse
from .runner import Runner

Transition = namedtuple('Transition', ('action_outs', 'actions', 'rewards', 'values', 'episode_masks', 'episode_agent_masks'))
class RunnerBaseline(Runner):
    def __init__(self, config, env, agent):
        super(RunnerBaseline, self).__init__(config, env, agent)


    def run_an_episode(self):

        memory = []
        info = dict()
        log = dict()
        # episode_return = np.zeros(self.n_agents)
        episode_return = 0

        self.reset()
        obs = self.env.get_obs()
        step = 1
        done = False
        while not done and step <= self.args.episode_length:
            if step == 1 and self.args.hard_attn and self.args.commnet:
                info['comm_action'] = np.zeros(self.args.n_agents, dtype=int)
            if self.args.commnet and self.args.recurrent:

                self.args.rnn_type = 'LSTM'
            if self.args.recurrent:
                if self.args.rnn_type == 'LSTM' and step == 1:
                    # prev_hid = self.policy_net.init_hidden(batch_size=state.shape[0])
                    prev_hid = self.agent.init_hidden(batch_size=1)

                obs_tensor = torch.tensor(np.array(obs), dtype=torch.float)
                obs_tensor = obs_tensor.unsqueeze(0)  #
                obs_tensor = [obs_tensor, prev_hid]  #

                action_outs, values, prev_hid = self.agent(obs_tensor, info)

                if (step) % self.args.detach_gap == 0:
                    if self.args.rnn_type == 'LSTM':
                        prev_hid = (prev_hid[0].detach(), prev_hid[1].detach())
                    else:
                        prev_hid = prev_hid.detach()

        #prev_hid = self.agent.init_hidden(batch_size=state.shape[0])

        # for t in range(self.args.episode_length):
        #     if t == 0 and self.args.hard_attn and self.args.commnet:
        #         info['comm_action'] = np.zeros(self.args.n_agents, dtype=int)
            else:
                obs_tensor = torch.tensor(np.array(obs), dtype=torch.float)
                action_outs, values = self.agent(obs_tensor,info)#

            actions = self.choose_action(action_outs)

            rewards, done, env_info = self.env.step(actions)

            next_obs = self.env.get_obs()
            if self.args.hard_attn and self.args.commnet:
                info['comm_action'] = actions[-1] if not self.args.comm_action_one else np.ones(self.args.n_agents,
                                                                                                dtype=int)

            # episode_mask = np.zeros(np.array(rewards).shape)
            # episode_agent_mask = np.array(dones) + 0
            #
            # if dones or t == self.args.episode_length - 1:
            # # if all(dones) or t == self.args.episode_length - 1:
            #     episode_mask = np.ones(np.array(rewards).shape)
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


            #state = next_state
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
            # if all(done) or t == self.args.episode_length - 1:
            #     log['episode_return'] = [episode_return]
            #     log['episode_steps'] = [t + 1]
            #     log['num_steps'] = t + 1
            #     break

        return memory ,log



