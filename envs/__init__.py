# from .multiagentenv import MultiAgentEnv, _GymWrapper
from .tj_wrappers import TJ_Wrapper
from .traffic_junction.traffic_junction_world import TrafficJunctionEnv
from .wrappers import Wrapper
from pathlib import Path
import os
from gym import register
from itertools import product
import gym
import itertools
from envs.rware.warehouse import Warehouse, RewardType, Action

envs = Path(os.path.dirname(os.path.realpath(__file__))).glob("**/*_v?.py")
for e in envs:
    name = e.stem.replace("_", "-")
    lib = e.parent.stem
    filename = e.stem
    gymkey = f"{lib}-{name}"
    register(
        gymkey,
        entry_point="envs.mpe:PettingZooWrapper",
        kwargs={"lib_name": lib, "env_name": filename,},
    )


register(
    id='TrafficJunction-v0',
    entry_point='envs.traffic_junction.traffic_junction_world:TrafficJunctionEnv',
)



register(
    id='PredatorPrey-v0',
    entry_point='envs.pp.predator_prey_env:PredatorPreyEnv',
)


sizes = range(5, 20)
players = range(2, 20)
foods = range(1, 10)
coop = [True, False]
partial_obs = [True]


# for s, p, f, c, po in product(sizes, players, foods, coop, partial_obs):
#     id = "Foraging{4}-{0}x{0}-{1}p-{2}f{3}-v2".format(s, p, f, "-coop" if c else "", "-2s" if po else "")
#     print(id)
# Foraging-2s- 10x10- 3player- 3food    -v2

# for s, p, f, c, po in product(sizes, players, foods, coop, partial_obs):
#     register(
#         id="Foraging{4}-{0}x{0}-{1}p-{2}f{3}-v2".format(s, p, f, "-coop" if c else "", "-2s" if po else ""),
#         entry_point="envs.lbforaging.foraging:ForagingEnv",
#         kwargs={
#             "players": p,
#             "max_player_level": 3,
#             "field_size": (s, s),
#             "max_food": f,
#             "sight": 2 if po else s,
#             "max_episode_steps": 50,
#             "force_coop": c,
#             "grid_observation": False,
#         },
#     )

Type = ['easy', 'medium', 'hard']
for t in Type:
    register(
        id="Foraging-{0}-v0".format(t),
        entry_point="envs.lbforaging.foraging:ForagingEnv",
        kwargs={
            'type':t
    },
    )





_sizes = {
    "tiny": (1, 3),
    "small": (2, 3),
    "medium": (2, 5),
    "large": (3, 5),
}

_difficulty = {"-easy": 2, "medium": 1, "-hard": 0.5}

_perms = itertools.product(_sizes.keys(), _difficulty, range(1, 20), range(1, 5))

for size, diff, agents, colors in _perms:
    # normal tasks
    gym.register(
        id=f"rware-{colors}color-{size}-{agents}ag{diff}-v1",
        entry_point="envs.rware.warehouse:Warehouse",
        kwargs={
            "column_height": 8,
            "shelf_rows": _sizes[size][0],
            "shelf_columns": _sizes[size][1],
            "n_agents": agents,
            "msg_bits": 0,
            "sensor_range": 1,
            "request_queue_size": int(agents * _difficulty[diff]),
            "max_inactivity_steps": None,
            "max_steps": 500,
            "reward_type": RewardType.INDIVIDUAL,
            "colors": colors,
        },
    )



REGISTRY = {}
REGISTRY["lbf"] = Wrapper
REGISTRY["rware"] = Wrapper
REGISTRY["tj"] = TJ_Wrapper
REGISTRY["mpe"] = Wrapper
REGISTRY["pp"] = Wrapper
