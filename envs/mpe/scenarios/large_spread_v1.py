import math

import numpy as np
from .._mpe_utils.core import World, Agent, Landmark
from .._mpe_utils.scenario import BaseScenario
import random
import matplotlib.pyplot as plt
import networkx as nx
from modules.graph import measure_strength

class Scenario(BaseScenario):
    def make_world(self, groups, cooperative=False, shuffle_obs=False):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = sum(groups)
        num_landmarks = len(groups)
        world.collaborative = True

        self.shuffle_obs = shuffle_obs

        self.num_agents = num_agents

        self.cooperative = cooperative
        self.groups = groups
        self.group_indices = [a * [i] for i, a in enumerate(self.groups)]
        self.group_indices = [
            item for sublist in self.group_indices for item in sublist
        ]

        

        # generate colors:
        self.colors = [np.random.random(3) for _ in groups]

        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = "agent_{}".format(i)
            agent.collide = False
            agent.silent = True
            agent.size = 0.15
            agent.group_id = self.group_indices[i]
            agent.group_one_hot =  [0] * num_landmarks
            agent.group_one_hot[agent.group_id] = 1
            agent.id = [0] * num_agents
            agent.id[i] = 1
            if agent.group_id == 0:
                agent.act = True
            elif agent.group_id == 1:
                agent.act = True
            else:
                agent.act = False
            

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.id = self.group_indices[i]
            landmark.collide = False
            landmark.movable = False
        return world





    def reset_world(self, world, np_random):
        # random properties for agents

        for i, agent in zip(self.group_indices, world.agents):
            agent.color = self.colors[i]

        # random properties for landmarks
        for landmark, color in zip(world.landmarks, self.colors):
            landmark.color = color

        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np_random.uniform(-3, +3, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [
                np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos)))
                for a in world.agents
            ]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions

        if not agent.act:
            return 0

        i = world.agents.index(agent)
        landmark_index = self.group_indices[i]
        rew = -np.sqrt(np.sum(np.square(agent.state.p_pos- world.landmarks[self.group_indices[i]].state.p_pos)))
        # rew = 0
        # for i, a in zip(self.group_indices, world.agents):
        #     if landmark_index == world.landmarks[i].id:
        #         rew -= np.sqrt(np.sum(np.square(a.state.p_pos - world.landmarks[i].state.p_pos)))

  
        if agent.collide:
            for a in world.agents:
                if a.act:
                    if self.is_collision(a, agent):
                        rew -= 5

        # if self.cooperative:
        #     return 0
        # else:
        return rew

    def global_reward(self, world):
        # rew = 0
        #
        # for i, a in zip(self.group_indices, world.agents):
        #     l = world.landmarks[i]
        #     rew -= np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos)))
        #
        # rew = rew / self.num_agents
        #
        # #if self.cooperative:
        # return rew
        # # else:
        return 0

    def observation(self, agent, world):
        entity_pos = []
        for  entity_index, entity in enumerate (world.landmarks):  # world.entities:
            related_pos = entity.state.p_pos - agent.state.p_pos

            if np.linalg.norm(related_pos) <= 2 or agent.group_id != entity_index:
                entity_pos.append(np.array(related_pos))
            else:
                entity_pos.append(np.array([20,20]))

        if agent.act:
                x = np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + [agent.group_one_hot])
        else:
                x = np.concatenate([np.zeros_like(agent.state.p_vel)] + [np.zeros_like(agent.state.p_pos)])

        if self.shuffle_obs:
            x = list(x)
            random.Random(self.group_indices[world.agents.index(agent)]).shuffle(x)
            x = np.array(x)
        return x