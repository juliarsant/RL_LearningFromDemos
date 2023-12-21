"""
Julia Santaniello
06/25/23

For training a single policy. Input hyperparamters and saved policy name.
"""
import numpy as np
import time
from PIL import Image
import pandas as pd
import pygame
from simplePG import SimplePG
from lunar_lander import LunarLander
import demonstrations as demo
import pickle
import matplotlib.pyplot as plt
import csv
from imports import demo_name_hitl, seed, steps, gamma, learning_rate, obs_size_values, num_actions,algorithm_name, env_name, episodes, num_demos, trials

env = LunarLander()
demo_name = demo_name_hitl

def start():
    input("Press Enter to Start Demonstrations: ")
    print("Starting in 3...")
    time.sleep(1)
    print("2...")
    time.sleep(1)
    print("1...")
    time.sleep(1)
    print("Start!")
    print("")

def run(trials):
    start()
    demo.main(demo_name)
    print("Thank you!")
    time.sleep(1)
    print("Starting agent training...")

    r,s,a=[],[],[]
    for i in range(trials):
        rewards, steps, accuracy = train_with_demonstrations()
        r.append(rewards)
        s.append(steps)
        a.append(accuracy)

    print("Done! Saving Data!")
    r = np.asarray(r).mean(axis=0)
    s = np.asarray(s).mean(axis=0)
    a = np.asarray(a).mean(axis=0)
    save_data(r, s, a)

def save_data(r,s,a):
    arr = np.asarray([r,s,a])
    df = pd.DataFrame(arr)
    df.to_csv("./data/results/{}.csv".format(demo_name))

        
def train_with_demonstrations():
    with open('./data/demonstrations/{}.pickle'.format(demo_name), 'rb') as file:
        demo_dict = pickle.load(file)

    file.close()
    env = LunarLander()
    agent = SimplePG(num_actions=num_actions, input_size=obs_size_values, hidden_layer_size=12, learning_rate=learning_rate, decay_rate=0.99, gamma=gamma, greedy_e_epsilon=0.1, random_seed=10)


    demo_eps = len(demo_dict)-2

    avg_rewards_past, avg_steps_past, avg_accuracy_past = [], [], []
    sum_wins, running_reward, running_steps = 0,0,0

    for i in range(episodes):
        ii = i-demo_eps
        if i < demo_eps:
            steps = demo_dict[i]["steps"]
            seed = demo_dict[i]["seed"]
            state = env.reset(seed=seed)
        
        else:
            steps = steps
            state = env.reset()

        episode_reward = 0

        for j in range(steps):
            #pick an action
            if i < num_demos:
                human_action = demo_dict[i]["actions"][j]
                agent.save_human_action(human_action)
                human_reward = demo_dict[i]["rewards"][j]
                human_state = demo_dict[i]["states"][j]
                agent.saveHumanStep(state=human_state, reward=human_reward, action=human_action)

            action = agent.pickAction(state, exploring=True)
            next_state, reward, done, win= env.step(action)
            agent.saveStep(state=state, reward=reward, action=action, next_state=next_state, done=done)

            episode_reward += reward
            state = next_state

            if done and ii>=0:
                running_steps += j
                running_reward += episode_reward
                if win:
                    sum_wins += 1
                break
            elif done:
                break
        
        if i < demo_eps:
            agent.finishEpisode(episode_reward, True)
        else:
            agent.finishEpisode(episode_reward, False)

        if ii % 20 == 0:
            avg_accuracy_past.append(sum_wins/20)
            avg_rewards_past.append(running_reward/20)
            avg_steps_past.append(running_steps/20)
            print('Episode {}\tlength: {}\treward: {}'.format(i, j, running_reward/20))
            running_reward, running_steps, sum_wins = 0,0,0

        agent.updateParamters()
    
    assert(len(avg_rewards_past)==len(avg_steps_past)==len(avg_accuracy_past))

    return avg_rewards_past, avg_steps_past, avg_accuracy_past


"""
Plots()
Purpose: Plot policy trained with or without demonstrations
Return: None; Print plots
"""
def plots(rewards_demos, rewards_no_demos):
    num_demos = len(rewards_demos) - len(rewards_no_demos)

    #X-axis
    iterations = range(0,len(rewards_no_demos)*20,20)

    #Plot
    plt.plot(iterations, rewards_demos[num_demos:])
    plt.plot(iterations, rewards_no_demos)
    plt.ylabel("Average Return")
    plt.xlabel("Iterations")
    plt.legend(["With Human", "Without Human"])
    plt.show()


        

if __name__ == "__main__":
    run(trials)