# BRAX -----------------------------------------------------------------------
from brax.training.agents.apg import train as apg

# Workspace --------------------------------------------------------------------
import anymal_brax.training_functions

# Hyperparameters -------------------------------------------------------------
def load_config() -> dict:
  config = {
    "result_indentifier": 'reacher', # no need for underscore
    "env_name": 'reacher',  # @param ['anymal_c', 'a1', 'ant', 'halfcheetah', 'hopper', 'humanoid', 'humanoidstandup', 'inverted_pendulum', 'inverted_double_pendulum', 'pusher', 'reacher', 'walker2d']
    "backend": 'positional',  # @param ['generalized', 'positional', 'spring']
    "training_agent": apg,
    "episode_length": 150, # @param {type: 'integer'}
    "truncation_length": 10, # @param {type: 'integer'}
    "learning_rate": 1e-3,  # @param {type: 'number'}, default: 3e-4
    "num_envs": 400,  # @param {type: 'integer'}
    "num_episodes": 10,  # @param {type: 'integer'}
    "action_repeat": 2,  # @param {type: 'integer'}
    "deterministic_eval": True,  # @param {type: 'boolean'}
  }
  return config

if __name__ == '__main__':
  config = load_config()
  anymal_brax.training_functions.train(config=config)