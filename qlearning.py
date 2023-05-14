import gymnasium as gym
import numpy as np
import random
import pickle


def train(episode_number, max_steps, alpha, gamma, epsilon, decay_rate, render_mode):
    env = gym.make("FrozenLake-v1", render_mode=render_mode, is_slippery=False)
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
            random_num = random.random()

            if random_num < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(qtable[state, :])

            if epsilon > 0.01:
                epsilon -= decay_rate

            # print(action)
            new_state, reward, done, info, prob = env.step(action)

            qtable[state, action] = alpha * (reward) + (1 - alpha) * (
                    qtable[state, action] + gamma * (np.max(qtable[new_state, :])))

            # print(qtable[state, action])

            if done:
                break
            state = new_state
    # print(qtable)
    save_train_result(qtable)

    env.close()


def exploit_trained_qtable(max_steps, alpha, gamma, render_mode):
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
    with open('qtable-qlearning.pickle', 'wb') as handle:
        pickle.dump(qtable, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_train_result():
    with open('qtable-qlearning.pickle', 'rb') as f:
        qtable = pickle.load(f)
    return qtable
