# frozenlake_RL_solutions
Here i put my implemention of reinforcement learning based solutions for the
<a href=https://gymnasium.farama.org/environments/toy_text/frozen_lake/>frozenlake</a> game.

- install required packages with
```
    pip install -r requirement.txt
```
## Q-learning
<a href=https://en.wikipedia.org/wiki/Q-learning/> q learning </a> is a model-free off-policy reinforcement learning algorithm.
- training q table :
first uncomment this line in the <a href=https://github.com/Mehrdadghassabi/frozenlake_RL_solutions/blob/main/main.py/> main.py </a>
```
    qlearning.train(episode_number, max_steps, alpha, gamma, epsilon, decay_rate, render_mode, desc)
```
then run <a href=https://github.com/Mehrdadghassabi/frozenlake_RL_solutions/blob/main/main.py/> main.py </a>
```
    python main.py
```
it takes few minutes because it is training the q-table using <a href=https://github.com/Mehrdadghassabi/frozenlake_RL_solutions/blob/main/qlearning.py/> qlearning.py </a>

## SARSA 
 <a href=https://en.wikipedia.org/wiki/State%E2%80%93action%E2%80%93reward%E2%80%93state%E2%80%93action/> Sarsa </a> is a model-free on-policy reinforcement learning algorithm.
- training q table :
same as the q-learning first uncomment this line in the <a href=https://github.com/Mehrdadghassabi/frozenlake_RL_solutions/blob/main/main.py/> main.py </a>
```
    Sarsa.train(episode_number, max_steps, alpha, gamma, epsilon, decay_rate, render_mode)
```
then run <a href=https://github.com/Mehrdadghassabi/frozenlake_RL_solutions/blob/main/main.py/> main.py </a>
```
    python main.py
```
it also takes few minutes because it is training the q-table using <a href=https://github.com/Mehrdadghassabi/frozenlake_RL_solutions/blob/main/Sarsa.py/> Sarsa.py </a>

## Monte-Carlo 
 we can also use <a href=https://en.wikipedia.org/wiki/State%E2%80%93action%E2%80%93reward%E2%80%93state%E2%80%93action/> Monte-Carlo method </a> for
 solving it.
- training q table :
same as the two last approach uncomment this line in the <a href=https://github.com/Mehrdadghassabi/frozenlake_RL_solutions/blob/main/main.py/> main.py </a>
```
    Monte_Carlo.train(episode_number, max_steps, epsilon, decay_rate, render_mode)
```
then run <a href=https://github.com/Mehrdadghassabi/frozenlake_RL_solutions/blob/main/main.py/> main.py </a>
```
    python main.py
```
it also takes few minutes because it is training the q-table using <a href=https://github.com/Mehrdadghassabi/frozenlake_RL_solutions/blob/main/Monte_Carlo.py/> Monte_Carlo.py </a>

## render mode
as training runs 100000 episode of the game its not wise to visualize all of them during the training, in order to do so
set the render mode to rgb_array in the <a href=https://github.com/Mehrdadghassabi/frozenlake_RL_solutions/blob/main/main.py/> main.py </a>
```
    render_mode = "rgb_array"
```
but if you want to see what happen set it to human
```
    render_mode = "human"
```
