from .traffic_junction_world import TrafficJunctionEnv

easy_dict = {
    'n_agents': 5,
    'dim': 6,
    'episode_length': 20,
    'add_rate_min': 0.3,
    'add_rate_max': 0.3,
    'difficulty': 'easy',
    'vision': 0,
    'vocab_type': 'bool',
    'curr_start': 0,
    'curr_end': 0}

medium_dict = {
    'n_agents': 10,
    'dim': 14,
    'episode_length': 40,
    'add_rate_min': 0.2,
    'add_rate_max': 0.2,
    'difficulty': 'medium',
    'vision': 0,
    'vocab_type': 'bool',
    'curr_start': 0,
    'curr_end': 0}


hard_dict = {
    'n_agents': 20,
    'dim': 18,
    'episode_length': 80,
    'add_rate_min': 0.05,
    'add_rate_max': 0.05,
    'difficulty': 'hard',
    'vision': 0,
    'vocab_type': 'bool',
    'curr_start': 0,
    'curr_end': 0}





