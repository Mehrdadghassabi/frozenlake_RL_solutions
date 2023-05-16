import qlearning
import Sarsa
import Monte_Carlo
from gym.envs.toy_text.frozen_lake import generate_random_map

max_steps = 100
alpha = 0.5
gamma = 0.5
epsilon = 0.5
decay_rate = 0.000001
episode_number = 100000
render_mode = "human"
proba_frozen = 0.7
map_size = 4
desc = generate_random_map(
    size=map_size, p=proba_frozen
)

# qlearning.train(episode_number, max_steps, alpha, gamma, epsilon, decay_rate, render_mode, desc)
# qlearning.exploit_trained_qtable(max_steps, render_mode)
# Sarsa.train(episode_number, max_steps, alpha, gamma, epsilon, decay_rate, render_mode)
# Sarsa.exploit_trained_qtable(max_steps, render_mode)
# Monte_Carlo.train(episode_number, max_steps, epsilon, decay_rate, render_mode)
# Monte_Carlo.exploit_trained_qtable(max_steps, render_mode)
