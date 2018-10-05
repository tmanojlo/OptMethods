import gym
import numpy as np
import matplotlib.pyplot as plt

def value_iteration(gamma, epsilon, states, actions, probabilities):
    U = np.zeros(env.nS)
    U_t = U.copy()

    while True:
        for state in range(states):
            temp_U = np.zeros(actions)
            for action in range(actions):
                for (prob, dst_state, reward, is_final) in probabilities[state][action]:
                    if(is_final == False):
                        temp_U[action] += prob*(reward + gamma*U_t[dst_state])
            U[state] = max(temp_U)
        print(np.sum(np.fabs(U-U_t)))
        diffs.append(np.sum(np.fabs(U-U_t)))
        if np.sum(np.fabs(U_t-U)) < epsilon:
            break
        U_t = U.copy()
    return U_t,diffs

def policy(states,actions,probabilities, U):
    result_policy = np.zeros(states)
    for state in range(states):
        profits = np.zeros(actions)
        for action in range(actions):
            for (prob, dst_state, reward, is_final) in probabilities[state][action]:
                profits[action] += prob*(reward + gamma*U[dst_state])
        result_policy[state] = np.argmax(profits)
    return result_policy


env = gym.make('Taxi-v2').unwrapped
observation = env.reset()
gamma = 0.999
cumulative_reward = 0

states = env.nS
actions = env.nA
probabilities = env.P

diffs = []

epsilon=1e-20
U,diffs = value_iteration(gamma, epsilon, states, actions, probabilities)


policy = policy(states,actions,probabilities, U).astype(np.int)
for t in range(1000):
    action = policy[observation]
    observation, reward, done, info = env.step(action)
    cumulative_reward += reward
    env.render()
    if done:
        print(t)
        break

plt.plot(diffs)
plt.title("$\gamma$ = " + str(gamma))
plt.ylabel('|$U(s) - U(s_{t+1}$)|')
plt.xlabel('Iteracija')
plt.show()
