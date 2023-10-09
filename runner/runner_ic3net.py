import torch
import torch.nn as nn
from collections import namedtuple
import numpy as np
from torch.optim import Adam

from modules.utils import merge_dict, multinomials_log_density
import time
import argparse
from .runner import Runner

Transition = namedtuple('Transition', ('action_outs', 'actions', 'rewards', 'values', 'episode_masks', 'episode_agent_masks'))
class RunnerIcnet(Runner):
    def __init__(self, config, env, agent):
        super(RunnerIcnet, self).__init__(config, env, agent)


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
            if step == 1:
                info['comm_action'] = np.zeros(self.args.n_agents, dtype=int)
            if self.args.recurrent:

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
                # action_outs = action_outs.unsqueeze(0)


            rewards, done, env_info = self.env.step(actions)

            next_obs = self.env.get_obs()

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


        if self.args.env == 'tj':
            merge_dict(self.env.get_stat(), log)
            # if all(done) or t == self.args.episode_length - 1:
            #     log['episode_return'] = [episode_return]
            #     log['episode_steps'] = [t + 1]
            #     log['num_steps'] = t + 1
            #     break

        return memory ,log




    def compute_grad(self, batch):
        return self.compute_agent_grad(batch)



    def compute_agent_grad(self, batch):

        log = dict()

        n = self.n_agents
        batch_size = len(batch.actions)
        rewards = torch.Tensor(batch.rewards)
        actions = torch.Tensor(batch.actions)
        actions = actions.transpose(1, 2)
        # actions = actions.reshape(-1, n, self.args.dim_actions)
        actions = actions.reshape(-1, n, 2)

        episode_masks = torch.Tensor(batch.episode_masks)
        episode_agent_masks = torch.Tensor(batch.episode_agent_masks)


        values = torch.cat(batch.values, dim=0)  # (batch, n, 1)
        action_outs = list(zip(*batch.action_outs))
        # action_outs = torch.Tensor(batch.action_outs)
        # # action_outs = batch.action_outs
        # action_outs = action_outs.transpose(1, 2).view(-1, n, 2)
        # for tnsr in action_outs:
        #     tnsr=list(tnsr)
        #     for b in tnsr:
        #         b=b.unsqueeze(0)
        #         c=tnsr
        action_outs = [torch.cat(a, dim=0) for a in action_outs]
        action_outs = [a.view(batch_size, -1, a.shape[1]) for a in action_outs]

        returns = torch.Tensor(batch_size, n)
        advantages = torch.Tensor(batch_size, n)
        values = values.view(batch_size, n)
        prev_returns = 0



        if self.args.normalize_rewards:
            mean_reward = rewards.mean()
            std_reward = rewards.std() + 1e-5  # Add a small value to prevent division by zero
            rewards = (rewards - mean_reward) / std_reward




        for i in reversed(range(rewards.size(0))):
            returns[i] = rewards[i] + self.args.gamma * prev_returns * episode_masks[i] * episode_agent_masks[i]
            prev_returns = returns[i].clone()



        for i in reversed(range(rewards.size(0))):
            advantages[i] = returns[i] - values.data[i]

        # if self.args.normalize_rewards:
        #     advantages = (advantages - advantages.mean()) / advantages.std()

        # element of log_p_a: [(batch_size*n) * num_actions[i]]
        # log_p_a = [action_outs.view(-1, self.n_actions)]
        # log_p_a = [a.view(-1, self.n_actions) for a in action_outs]
        num_actions=[self.n_actions, 2]
        log_p_a = [action_outs[i].view(-1, num_actions[i]) for i in range(2)]
        # actions: [(batch_size*n) * dim_actions]
        actions = actions.contiguous().view(-1, 2)
        log_prob = multinomials_log_density(actions, log_p_a)
        action_loss = -advantages.view(-1) * log_prob.squeeze()
        actor_loss = action_loss.sum()


        targets = returns
        value_loss = (values - targets).pow(2).view(-1)
        critic_loss = value_loss.sum()


        total_loss = actor_loss + self.args.value_coeff * critic_loss
        total_loss.backward()


        log['action_loss'] = actor_loss.item()
        log['value_loss'] = critic_loss.item()
        log['total_loss'] = total_loss.item()

        return log



    def choose_action(self, log_p_a):
        log_p_a = [a.unsqueeze(0) for a in log_p_a]
        p_a = [[z.exp() for z in x] for x in log_p_a]
        ret = torch.stack([torch.stack([torch.multinomial(x, 1).detach() for x in p]) for p in p_a])
        action = [x.squeeze().data.numpy() for x in ret]
        return action