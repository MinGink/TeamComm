from ._mpe_utils.simple_env import SimpleEnv, make_env
from .scenarios.large_spread import Scenario
from utils.to_parallel import parallel_wrapper_fn


class raw_env(SimpleEnv):
    def __init__(self, seed=None, local_ratio=0.5, max_frames=100, **env_args):
        assert (
            0.0 <= local_ratio <= 1.0
        ), "local_ratio is a proportion. Must be between 0 and 1."
        scenario = Scenario()
        world = scenario.make_world(groups=[5, 5, 5])
        super().__init__(scenario, world, max_frames, local_ratio)


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)
