"""
Julia Santaniello
06/25/23

For training a single policy. Input hyperparamters and saved policy name.
"""
import numpy as np
import time
from PIL import Image
import pandas as pd
import csv
import pygame
from simplePG import SimplePG
from lunar_lander import LunarLander
from imports import demo_name, seed, steps, gamma, learning_rate, obs_size_values, num_actions,algorithm_name, env_name, episodes

env = LunarLander()
agent = SimplePG(num_actions=num_actions, input_size=obs_size_values, hidden_layer_size=12, learning_rate=learning_rate, decay_rate=0.99, gamma=gamma, greedy_e_epsilon=0.1, random_seed=10)
demo_name="train"
"Train an agent without demonstration"

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

def run_train(trials):
    # start()
    # demo.main(demo_name)
    print("Thank you!")
    time.sleep(1)
    print("Starting agent training...")

    r,s,a=[],[],[]
    for i in range(trials):
        rewards, steps, accuracy = train()
        r.append(rewards)
        s.append(steps)
        a.append(accuracy)
    
    r = np.asarray(r).mean(axis=0)
    s = np.asarray(s).mean(axis=0)
    a = np.asarray(a).mean(axis=0)

    print("Done! Saving Data!")
    save_data(r, s, a)


def train():
    avg_rewards_past, avg_steps_past, avg_accuracy_past = [],[],[]
    state = env.reset(seed=10)
    sum_wins, running_reward, running_steps = 0,0,0


    for i_episode in range(0, episodes):
        state = env.reset()
        episode_rewards = 0
        next_state=None

        for t in range(steps):
            action = agent.pickAction(state, exploring=True)
            next_state, reward, done, win = env.step(action)
            episode_rewards += reward
            agent.saveStep(state=state, reward=reward, action=action, next_state=next_state, done=done)
            
            state = next_state

            if done:
                running_steps += t
                running_reward += episode_rewards
                if win:
                    sum_wins+=1
                break
        
        # Updating the policy :
        agent.finishEpisode()

        if i_episode % 20 == 0:
            avg_accuracy_past.append(sum_wins/20)
            avg_rewards_past.append(running_reward/20)
            avg_steps_past.append(running_steps/20)
            print('Episode {}\tlength: {}\treward: {}'.format(i_episode, t, running_reward/20))
            running_reward, running_steps, sum_wins = 0,0,0

        agent.updateParamters()

    assert(len(avg_rewards_past)==len(avg_steps_past)==len(avg_accuracy_past))

    return avg_rewards_past, avg_steps_past, avg_accuracy_past

def save_data(r,s,a):
    arr = np.asarray([r,s,a])
    df = pd.DataFrame(arr)
    df.to_csv("./data/results/{}.csv".format(demo_name))

if __name__ == "__main__":
    run_train()


#random_seed = 543
#torch.manual_seed(random_seed)
#env = gym.make('LunarLander-v2')
#env.seed(random_seed)
    
# if average_reward > 210:
#     torch.save(policy.state_dict(), './preTrained/LunarLander_{}.pth'.format(title))
#     print("########## Solved! ##########")
#     #test(name='LunarLander_{}}.pth')#.format(title))
#     break