import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import argparse


class TeamCommAgent(nn.Module):

    def __init__(self, agent_config):
        super(TeamCommAgent, self).__init__()

        self.args = argparse.Namespace(**agent_config)
        self.seed = self.args.seed

        self.n_agents = self.args.n_agents
        self.hid_size = self.args.hid_size

        self.agent = AgentAC(self.args)
        self.teaming = Teaming(self.args)


        self.block = self.args.block





    def communicate(self, obs, sets):

        local_obs = self.agent.local_emb(obs)

        #comm_obs = self.agent.comm_emb(obs)
        inter_obs_emb = self.agent.inter_emb(obs)
        intra_obs_emb = self.agent.intra_emb(obs)

        inter_obs = torch.zeros_like(inter_obs_emb)
        intra_obs = torch.zeros_like(intra_obs_emb)


        #do attention for each set, and then concat
        inter_mu = torch.zeros_like(inter_obs)
        inter_std = torch.zeros_like(inter_obs)
        intra_mu = torch.zeros_like(intra_obs)
        intra_std = torch.zeros_like(intra_obs)


        global_set = []
        for set in sets:
            member_obs = intra_obs_emb[set,:]
            member_obs_pooling_input = inter_obs_emb[set,:]
            intra_obs[set,:], intra_mu[set,:], intra_std[set,:] = self.agent.intra_com(member_obs)
            pooling = self.agent.pooling(member_obs_pooling_input)
            global_set.append(pooling)

        inter_obs_input = torch.cat(global_set, dim=0)
        inter_obs_output, inter_mu_output, inter_std_output = self.agent.inter_com(inter_obs_input)

        for index, set in enumerate(sets):
            if len(set) > 1:
                inter_obs[set,:] = inter_obs_output[index,:].repeat(len(set), 1)
                inter_mu[set,:] = inter_mu_output[index,:].repeat(len(set), 1)
                inter_std[set,:] = inter_std_output[index,:].repeat(len(set), 1)
            else:
                inter_obs[set,:] = inter_obs_output[index,:]
                inter_mu[set,:] = inter_mu_output[index,:]
                inter_std[set,:] = inter_std_output[index,:]



        if self.block == 'no':
            after_comm = torch.cat((local_obs,  inter_obs,  intra_obs), dim=-1)
        elif self.block == 'inter':
            after_comm = torch.cat((local_obs,  intra_obs, torch.rand_like(inter_obs)), dim=-1)
        elif self.block == 'intra':
            after_comm = torch.cat((local_obs,  inter_obs, torch.rand_like(intra_obs)), dim=-1)
        else:
            raise ValueError('block must be one of no, inter, intra')


        mu = torch.cat((intra_mu, inter_mu), dim=-1)
        std = torch.cat((intra_std, inter_std), dim=-1)

        return after_comm, mu, std







class AgentAC(nn.Module):
    def __init__(self, args):
        super(AgentAC, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.hid_size = args.hid_size
        self.n_actions = self.args.n_actions
        self.tanh = nn.Tanh()
        self.att_head = self.args.att_head

        self.message_dim = 64

        self.fc_1 = nn.Linear(self.hid_size +  self.message_dim * 2 , self.hid_size)
        self.fc_2 = nn.Linear(self.hid_size, self.hid_size)
        self.actor_head = nn.Linear(self.hid_size, self.n_actions)
        self.value_head = nn.Linear(self.hid_size, 1)




        self.local_fc_emb = nn.Linear(self.args.obs_shape, self.hid_size)
        self.inter_fc_emb = nn.Linear(self.args.obs_shape, self.message_dim)
        self.intra_fc_emb = nn.Linear(self.args.obs_shape, self.message_dim)



        self.intra_attn_mu = nn.MultiheadAttention(self.message_dim, num_heads=self.att_head, batch_first=True)
        self.intra_attn_std = nn.MultiheadAttention(self.message_dim, num_heads=self.att_head, batch_first=True)

        self.inter_attn_mu = nn.MultiheadAttention(self.message_dim, num_heads=self.att_head, batch_first=True)
        self.inter_attn_std = nn.MultiheadAttention(self.message_dim, num_heads=self.att_head, batch_first=True)

        self.attset_fc = nn.Linear(self.message_dim, 1)



    def forward(self, final_obs):
        h = self.tanh(self.fc_1(final_obs))
        h = self.tanh(self.fc_2(h))
        a = F.log_softmax(self.actor_head(h), dim=-1)
        v = self.value_head(h)
        return a, v


    def local_emb(self, x):
        return self.tanh(self.local_fc_emb(x))

    def inter_emb(self, x):
        return self.tanh(self.inter_fc_emb(x))

    def intra_emb(self, x):
        return self.tanh(self.intra_fc_emb(x))






    def intra_com(self, input):
        x = input.unsqueeze(0)
        mu, _ = self.intra_attn_mu(x,x,x)
        std, _ = self.intra_attn_std(x,x,x)
        std = F.softplus(std.squeeze(0)-5, beta = 1)
        intra_obs = self.reparameterise(mu.squeeze(0), std)
        return intra_obs, mu.squeeze(0), std



    def inter_com(self, input):
        x = input.unsqueeze(0)
        mu, _ = self.inter_attn_mu(x,x,x)
        std, _ = self.inter_attn_std(x,x,x)
        std = F.softplus(std.squeeze(0)-5, beta = 1)

        inter_obs = self.reparameterise(mu.squeeze(0), std)
        return inter_obs, mu.squeeze(0), std



    def pooling(self, input):
        score = F.softmax(self.attset_fc(input), dim=0)

        #zip the input to 1 * fixed size output based on score
        output = torch.sum(score * input, dim=0, keepdim=True)    # [1, 1, hid_size]
        return output




    def reparameterise(self, mu, std):
        eps = torch.randn_like(std)
        return mu + std * eps




class Teaming(nn.Module):

    def __init__(self, args):
        super(Teaming, self).__init__()

        self.args = args

        self.n_agents = self.args.n_agents
        self.max_group = 4

        self.hid_size = self.args.hid_size

        self.tanh = nn.Tanh()

        self.fc1 = nn.Linear(self.args.obs_shape * self.n_agents, self.hid_size * 2)
        self.fc2 = nn.Linear(self.hid_size * 2, self.hid_size)
        self.fc3 = nn.Linear(self.args.obs_shape, self.hid_size)
        self.fc4 = nn.Linear(self.hid_size * 2, self.hid_size)
        self.action_head = nn.Linear(self.hid_size, self.max_group)



        self.critic_fc1 = nn.Linear(self.args.obs_shape * self.n_agents, self.hid_size * 2)
        self.critic_fc2 = nn.Linear(self.hid_size * 2, self.hid_size)
        self.critic_fc3 = nn.Linear(self.n_agents, self.hid_size)
        self.critic_fc4 = nn.Linear(self.hid_size * 2, self.hid_size)
        self.value_head = nn.Linear(self.hid_size, 1)



    def forward(self, x):

        h = x.view(1, -1)
        h = self.tanh(self.fc1(h))
        z = self.tanh(self.fc2(h))
        x = self.tanh(self.fc3(x))
        xh = torch.cat([x, z.repeat(self.n_agents, 1)], dim=-1)
        xh= self.tanh(self.fc4(xh))
        a = F.log_softmax(self.action_head(xh), dim=-1)

        return a


    def critic(self, o, a):


        h = o.view(1, -1)
        h = self.tanh(self.critic_fc1(h))
        z = self.tanh(self.critic_fc2(h))

        a = torch.Tensor(np.array(a)).view(1, -1)
        a = self.tanh(self.critic_fc3(a))

        ha = torch.cat([z, a], dim=-1)
        ha = self.tanh(self.critic_fc4(ha))
        v = self.value_head(ha)
        return v