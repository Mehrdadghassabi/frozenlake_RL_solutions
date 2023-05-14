import qlearning
import Sarsa

max_steps = 100
alpha = 0.5
gamma = 0.5
epsilon = 0.5
decay_rate = 0.000001
episode_number = 100000
render_mode = "human"

# qlearning.train(episode_number, max_steps, alpha, gamma, epsilon, decay_rate, render_mode)
# qlearning.exploit_trained_qtable(max_steps, alpha, gamma, render_mode)
# Sarsa.train(episode_number, max_steps, alpha, gamma, epsilon, decay_rate, render_mode)
Sarsa.exploit_trained_qtable(max_steps, alpha, gamma, render_mode)
