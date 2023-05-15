import gymnasium as gym
import numpy as np
import random
import pickle


def train(episode_number, max_steps, epsilon, decay_rate, render_mode, desc):
    env = gym.make("FrozenLake-v1", render_mode=render_mode, is_slippery=False, desc=desc)
    env.reset()
    qtable = np.zeros((env.observation_space.n, env.action_space.n))
    Visit = np.zeros((env.observation_space.n, env.action_space.n))

    for episode in range(episode_number):
        state = env.reset()[0]
        done = False
        G = 0
        visted_state = []
        print("****************************************************")
        print("EPISODE ", episode)
        print("EPSILON ", epsilon)
        # print(qtable)
        for step in range(max_steps):
            env.render()

            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(qtable[state, :])

            if epsilon > 0.01:
                epsilon -= decay_rate

            visted_state.append((state, action))
            new_state, reward, done, info, prob = env.step(action)
            G += reward

            if done:
                break
            state = new_state

        for (state, action) in visted_state:
            Visit[state, action] += 1.0
            alpha = 1.0 / Visit[state, action]
            qtable[state, action] += alpha * (G - qtable[state, action])
    # print(qtable)
    save_train_result(qtable)

    env.close()


def exploit_trained_qtable(max_steps, render_mode):
    env = gym.make("FrozenLake-v1", render_mode=render_mode, is_slippery=False)
    state = env.reset()[0]
    qtable = get_train_result()
    done = False
    # print(qtable)
    for step in range(max_steps):
        env.render()

        action = np.argmax(qtable[state, :])

        new_state, reward, done, info, prob = env.step(action)

        # print(qtable[state, action])

        if done:
            break
        state = new_state
    env.close()


def save_train_result(qtable):
    with open('qtable-montecarlo.pickle', 'wb') as handle:
        pickle.dump(qtable, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_train_result():
    with open('qtable-montecarlo.pickle', 'rb') as f:
        qtable = pickle.load(f)
    return qtable
