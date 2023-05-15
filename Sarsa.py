import gymnasium as gym
import numpy as np
import random
import pickle


def train(episode_number, max_steps, alpha, gamma, epsilon, decay_rate, render_mode, desc):
    env = gym.make("FrozenLake-v1", render_mode=render_mode, is_slippery=False, desc=desc)
    env.reset()
    qtable = np.zeros((env.observation_space.n, env.action_space.n))

    for episode in range(episode_number):
        state = env.reset()[0]
        done = False
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

            # print(action)
            new_state, reward, done, info, prob = env.step(action)

            if random.random() < epsilon:
                action2 = env.action_space.sample()
            else:
                action2 = np.argmax(qtable[new_state, :])

            qtable[state, action] = alpha * (reward) + (1 - alpha) * (
                    qtable[state, action] + gamma * (qtable[new_state, action2]))

            if epsilon > 0.01:
                epsilon -= decay_rate

            # print(qtable[state, action])

            if done:
                break
            state = new_state
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
    with open('qtable-sarsa.pickle', 'wb') as handle:
        pickle.dump(qtable, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_train_result():
    with open('qtable-sarsa.pickle', 'rb') as f:
        qtable = pickle.load(f)
    return qtable
