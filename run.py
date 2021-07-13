# -*- coding: utf-8 -*-

import data_generator as gen
import ca_unet as unet
import tensorflow as tf
import random
import argparse, pathlib

parser = argparse.ArgumentParser(description='Generalization CA')
parser.add_argument('--gen_level', help='Level of Generalization, try "simple", "1", "2", or "extra", "inter". See papr for eplanation')
args = parser.parse_args()

if (args.gen_level == "simple"):
    import params.params_simple as params
elif (args.gen_level == "1"):
    import params.params_level_1 as params
elif (args.gen_level == "2"):
    import params.params_level_2 as params
elif (args.gen_level == "extra"):
    import params.params_level_3_extra as params
elif (args.gen_level == "inter"):
    import params.params_level_3_inter as params


# set grid parameters
size = params.grid_size
initial_alive = params.initial_alive
timesteps = params.timesteps

# generate the rules
total_rules = params.total_rules

train_rules = []
test_rules = []

# set the Moore neighborhood sizes used for data generation (train and test set)
neighborhood_sizes_train = params.neighborhood_sizes_train
neighborhood_sizes_test = params.neighborhood_sizes_test

# check for any duplicate rules in the train rules
print("Duplicate rules in train rules?")
print(any(train_rules.count(x) > 1 for x in train_rules))

# generate the trajectories with the specified rulesets
for i in range(total_rules):
    train_rules.append(gen.generate_rule(random.choice(neighborhood_sizes_train)))

for i in range(int(total_rules*0.1)):
    # 10% of training rules size in the test set
    test_rules.append(gen.generate_rule(random.choice(neighborhood_sizes_test)))

# in our case we use the same rules for validation as for training
val_rules = train_rules

# set the number of training, validation and test states
diff_states_train = params.diff_states_train
diff_states_val = int(diff_states_train*0.2)  # 20% validation
diff_states_test = int(diff_states_train*0.1) # 10% test

# generate the data
if (params.same_rules):
    test_rules = train_rules
    

X_train, y_train, X_val, y_val, X_test, y_test, train_rules_vector, val_rules_vector, test_rules_vector = gen.generate_data(train_rules, val_rules, test_rules, diff_states_train, diff_states_val, diff_states_test, size, initial_alive, timesteps)

# set training parameters
batch_size = params.batch_size
epochs = params.epochs
verbose = params.verbose

# create the model
model = unet.CA_Unet(size, timesteps)

# creat a checkpoint so the model saves the best parameters
mcp_save = tf.keras.callbacks.ModelCheckpoint('optimal_weights.hdf5', save_best_only=True, monitor='val_loss', mode='auto')

# train the model
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val), callbacks=[mcp_save], verbose = 2)

# evaluate the model on the different data sets
model.load_weights('optimal_weights.hdf5')

print("Performance on train set:")
model.evaluate(X_train,y_train, batch_size=batch_size)
print("Performance on validation set:")
model.evaluate(X_val,y_val, batch_size=batch_size)
print("Performance on test set:")
model.evaluate(X_test,y_test, batch_size=batch_size)
