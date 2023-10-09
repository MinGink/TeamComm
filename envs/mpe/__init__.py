from .utils.env import AECEnv
from gym.core import Env
from gym.spaces import Box, Discrete, Tuple
import importlib


class PettingZooWrapper(Env):

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, lib_name, env_name, **kwargs):

        PZEnv = importlib.import_module(f"envs.{lib_name}.{env_name}")
        #print(PZEnv)
        self._env = PZEnv.parallel_env(**kwargs)

        n_agents = self._env.num_agents

        self.action_space = Tuple(
            tuple([self._env.action_spaces[k] for k in self._env.agents])
        )
        self.observation_space = Tuple(
            tuple([self._env.observation_spaces[k] for k in self._env.agents])
        )

        self.n_agents = n_agents

    def reset(self):
        obs = self._env.reset()
        obs = tuple([obs[k] for k in self._env.agents])
        return obs

    def render(self, mode="human"):
        return self._env.render(mode)

    def step(self, actions):
        dict_actions = {}
        for agent, action in zip(self._env.agents, actions):
            dict_actions[agent] = action

        observations, rewards, dones, infos = self._env.step(dict_actions)

        obs = tuple([observations[k] for k in self._env.agents])
        rewards = [rewards[k] for k in self._env.agents]
        dones = [dones[k] for k in self._env.agents]
        info = {}
        return obs, rewards, dones, info

    def close(self):
        return self._env.close()


    def get_graph(self):
        return self._env.get_graph()

if __name__ == '__main__':
    print('test')