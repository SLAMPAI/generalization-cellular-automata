# -*- coding: utf-8 -*-

# imports
import numpy as np
import random
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

# functions
def make_environment(size, initial_alive):
    # create the grid the CA lives on as a matrix of 1 and 0
    # size: size of the grid
    # initial_alive: probability of a cell being alive
    return np.reshape(np.array([np.random.choice([0,1], size*size, p=[1-initial_alive, initial_alive])]), (size,size))

def update_environment(env, birth, stay, neighborhood_size):
    # compute the next timestep of an environment with given rules
    # env: the CA grid
    # birth, stay: lists with the numbers for the rules
    # neighborhood_size: size of the Moore neighborhood
    shape = env.shape
    neighbors = np.zeros(shape)
    # compute the number of living neighbors for each cell
    neighbors = convolve2d(env, np.ones((neighborhood_size, neighborhood_size)), mode='same', boundary='fill') - env
    
    # update the grid according to the rules (taken from Jenias code, faster with numpy operations)
    env_new = np.zeros(shape)
    env_new[np.where(np.in1d(neighbors, stay).reshape(shape) & (env == 1))] = 1
    env_new[np.where(np.in1d(neighbors, birth).reshape(shape) & (env == 0))] = 1

    return env_new

def compute_trajectory(size, inital_alive, steps, birth, stay, neighborhood_size):
    # compute multiple timesteps of a CA trajectory
    # size: size of the grid
    # initial_alive: probability of a cell being alive
    # steps: number of timesteps
    # birth, stay: lists with the numbers for the rules
    # neighborhood_size: size of the Moore neighborhood
    trajectory = np.zeros((size, size, steps))
    trajectory[:,:,0] = make_environment(size, inital_alive)
    for i in range(1, steps):
      trajectory[:,:,i] = update_environment(trajectory[:,:,i-1], birth, stay, neighborhood_size)
    return trajectory

def visualize(env, timesteps):
    # visualize a trajectory
    # env: the data
    # timesteps: the number of timesteps to visualize
    plt.figure(figsize=(8, 6), dpi=80)
    for i in range(timesteps):
      plt.subplot(1, timesteps, i+1)
      plt.title("t = %i" %i)
      plt.imshow(env[:,:,i], cmap='gray')

def generate_rule(neighborhood_size):
    # randomly pick the number of cells needed for birth and survival
    # neighborhood_size: size of the Moore neighborhood
    num_birth, num_stay = random.randint(1,neighborhood_size**2-1), random.randint(1,neighborhood_size**2-1)
    birth = (random.sample(range(1, neighborhood_size**2), num_birth))
    stay =  (random.sample(range(1, neighborhood_size**2), num_stay))
    return([birth, stay, neighborhood_size])

def generate_data(train_rules, val_rules, test_rules, diff_states_train, diff_states_val, diff_states_test, size, initial_alive, timesteps):
    # generate training, validation and test data
    # train_rules, val_rules, test_rules: the rule sets used for computing the trajectories
    # diff_states_train, diff_states_val, diff_states_test: number of states
    # size: size of the grid
    # initial_alive: probability of a cell being alive
    # timesteps: number of timesteps to be computed

    X_train = np.zeros((diff_states_train, size, size, timesteps))
    X_val = np.zeros((diff_states_val, size, size, timesteps))
    X_test = np.zeros((diff_states_test, size, size, timesteps))

    y_train = np.zeros((diff_states_train, size, size))
    y_val = np.zeros((diff_states_val, size, size))
    y_test = np.zeros((diff_states_test, size, size))


    train_rules_vector = []
    val_rules_vector = []
    test_rules_vector = []

    # generate trainig/validation/test data (X,y)
    # "stack the data"
    print("Generating training data... ")
    for i in range(0,diff_states_train):
          # sample a random rule
          [birth, stay, neighborhood_size] = random.choice(train_rules)

          # compute trajectory with the rule
          X_train[i,:,:,0:timesteps] = compute_trajectory(size, initial_alive, timesteps, birth, stay, neighborhood_size)
          y_train[i,:,:] = update_environment(X_train[i,:,:,timesteps-1], birth, stay, neighborhood_size)

          # save the rule used for the trajectory to a vector
          train_rules_vector.append([birth, stay, neighborhood_size])
    print("Done generating training data!")
           
    print("Generating validation data... ")
    # take rules from val set
    for i in range(0,int(diff_states_val)):
          # sample a random rule
          [birth, stay, neighborhood_size] = random.choice(val_rules)

          # compute trajectory with the rule
          X_val[i,:,:,0:timesteps] = compute_trajectory(size, initial_alive, timesteps, birth, stay, neighborhood_size)
          y_val[i,:,:] = update_environment(X_val[i,:,:,timesteps-1], birth, stay, neighborhood_size)

          # save the rule used for the trajectory to a vector
          val_rules_vector.append([birth, stay, neighborhood_size])
    print("Done generating validation data!")

    print("Generating test data... ")
    # take rules from test set
    for i in range(0,int(diff_states_test)):
        # sample a random rule
        [birth, stay, neighborhood_size] = random.choice(test_rules)

        # compute trajectory with the rule
        X_test[i,:,:,0:timesteps] = compute_trajectory(size, initial_alive, timesteps, birth, stay, neighborhood_size)
        y_test[i,:,:] = update_environment(X_test[i,:,:,timesteps-1], birth, stay, neighborhood_size)

        # save the rule used for the trajectory to a vector
        test_rules_vector.append([birth, stay, neighborhood_size])
    print("Done generating test data!")

    return X_train, y_train, X_val, y_val, X_test, y_test, train_rules_vector, val_rules_vector, test_rules_vector


