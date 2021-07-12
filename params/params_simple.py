# -*- coding: utf-8 -*-

# setting the data generation parameters

grid_size = 32
initial_alive = 0.2
timesteps = 4

total_rules = 100
neighborhood_sizes_train = [3]
neighborhood_sizes_test = [3]
same_rules = True
diff_states_train = 10000

# setting the network training parameters
batch_size = 256
epochs = 100
verbose = 2
