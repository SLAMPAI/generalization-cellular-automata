# -*- coding: utf-8 -*-

# imports
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, activations, optimizers, callbacks, metrics, backend, utils, initializers, regularizers


## check for gpu
if tf.test.gpu_device_name():
    print('Using GPU: {}'.format(tf.test.gpu_device_name()))
else:
    print("No GPU detected")

# ResNet Skip-Connections implementation
# (taken in part from https://github.com/marcopeix/Deep_Learning_AI/blob/master/4.Convolutional%20Neural%20Networks/2.Deep%20Convolutional%20Models/Residual%20Networks.ipynb)

def identity_block(X, f, filters):
    # X: input from the previous layer
    # f: kernel size
    # filters: number of filters used for the different layers within this block

    # identity blocks
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X
    
    # First component of main path
    X = layers.Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', kernel_initializer = initializers.glorot_uniform(), kernel_regularizer=regularizers.l2(0.0005))(X)
    X = layers.BatchNormalization(axis = 3)(X)
    X = layers.Activation('relu')(X)
    
    # Second component of main path
    X = layers.Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', kernel_initializer = initializers.glorot_uniform(), kernel_regularizer=regularizers.l2(0.0005) )(X)
    X = layers.BatchNormalization(axis = 3)(X)
    X = layers.Activation('relu')(X)

    # Third component of main path
    X = layers.Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', kernel_initializer = initializers.glorot_uniform(), kernel_regularizer=regularizers.l2(0.0005) )(X)
    X = layers.BatchNormalization(axis = 3)(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = layers.Add()([X, X_shortcut])
    X = layers.Activation('relu')(X)
    
    return X

def convolutional_block(X, f, filters, s=2):
    # X: input from the previous layer
    # f: kernel size
    # filters: number of filters used for the different layers within this block
    # s: stride used in the first conv layer

    # conv blocks

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path
    X = layers.Conv2D(filters = F1, kernel_size = (1, 1), strides = (s,s), padding = 'valid', kernel_initializer = initializers.glorot_uniform(), kernel_regularizer=regularizers.l2(0.0005))(X)
    X = layers.BatchNormalization(axis=3)(X)
    X = layers.Activation('relu')(X)


    # Second component of main path
    X = layers.Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', kernel_initializer = initializers.glorot_uniform(), kernel_regularizer=regularizers.l2(0.0005))(X)
    X = layers.BatchNormalization(axis=3)(X)
    X = layers.Activation('relu')(X)

    # Third component of main path
    X = layers.Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', kernel_initializer = initializers.glorot_uniform(), kernel_regularizer=regularizers.l2(0.0005))(X)
    X = layers.BatchNormalization(axis=3)(X)

    ##### SHORTCUT PATH ####
    X_shortcut = layers.Conv2D(filters = F3, kernel_size = (1, 1), strides = (s,s), padding = 'valid', kernel_initializer = initializers.glorot_uniform(), kernel_regularizer=regularizers.l2(0.0005))(X_shortcut)
  
    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = layers.Add()([X, X_shortcut])
    X = layers.Activation('relu')(X)

    return X


def identity_block_inv(X, f, filters):
    # X: input from the previous layer
    # f: kernel size
    # filters: number of filters used for the different layers within this block

    # inverse identiy blocks
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X

    # Third component of main path
    X = layers.Conv2DTranspose(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', kernel_initializer = initializers.glorot_uniform(), kernel_regularizer=regularizers.l2(0.0005))(X)
    X = layers.BatchNormalization(axis = 3)(X)
    X = layers.Activation('relu')(X)

    # Second component of main path
    X = layers.Conv2DTranspose(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', kernel_initializer = initializers.glorot_uniform(), kernel_regularizer=regularizers.l2(0.0005))(X)

    X = layers.BatchNormalization(axis = 3)(X)
    X = layers.Activation('relu')(X)

    # First component of main path
    X = layers.Conv2DTranspose(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', kernel_initializer = initializers.glorot_uniform(), kernel_regularizer=regularizers.l2(0.0005))(X)
    X = layers.BatchNormalization(axis = 3)(X)
    

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = layers.Add()([X, X_shortcut])
    X = layers.Activation('relu')(X)
    
    return X

def convolutional_block_inv(X, f, filters, s=2):
    # X: input from the previous layer
    # f: kernel size
    # filters: number of filters used for the different layers within this block
    # s: stride used in the first transpose-conv layer

    # inverse conv blocks

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####

    # Third component of main path
    X = layers.Conv2DTranspose(filters = F3, kernel_size = (1, 1), strides = (s,s), padding = 'valid', kernel_initializer = initializers.glorot_uniform(), kernel_regularizer=regularizers.l2(0.0005))(X)
    X = layers.BatchNormalization(axis=3)(X)
    X = layers.Activation('relu')(X)

    # Second component of main path
    X = layers.Conv2DTranspose(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', kernel_initializer = initializers.glorot_uniform(), kernel_regularizer=regularizers.l2(0.0005))(X)
    X = layers.BatchNormalization(axis=3)(X)
    X = layers.Activation('relu')(X)


    # First component of main path
    X = layers.Conv2DTranspose(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', kernel_initializer = initializers.glorot_uniform(), kernel_regularizer=regularizers.l2(0.0005))(X)
    X = layers.BatchNormalization(axis=3)(X)


    ##### SHORTCUT PATH ####
    X_shortcut = layers.Conv2DTranspose(filters = F1, kernel_size = (1, 1), strides = (s,s), padding = 'valid', kernel_initializer = initializers.glorot_uniform(), kernel_regularizer=regularizers.l2(0.0005))(X_shortcut)
  
    X_shortcut = layers.BatchNormalization(axis=3)(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = layers.Add()([X, X_shortcut])
    X = layers.Activation('relu')(X)

    return X

def CA_Unet(size, timesteps):
    #define the Unet with skip-connections
    # size: size of the grid/environment
    # timesteps: number of timesteps in the input data
    
    input_size = (size,size,timesteps)

    # input layer
    inputs = layers.Input(input_size)

    # Stage 1 down
    X = layers.Conv2D(32, (8, 8), strides = (2, 2), kernel_initializer = initializers.glorot_uniform(), padding='same', kernel_regularizer=regularizers.l2(0.0005))(inputs)
    X = layers.BatchNormalization(axis = 3)(X)
    X = layers.Activation('relu')(X)
    
    # Stage 2 down
    X = convolutional_block(X,f = 4, filters = [32, 32, 64], s = 1)
    X = identity_block(X, 4, [32, 32, 64])
    
    # Stage 3 down
    X = convolutional_block(X, f=2, filters=[64, 64, 128], s=2)
    X = identity_block(X, 2, [64, 64, 128])

    # latent space
    X = layers.Conv2D(filters = 256, kernel_size = (1, 1), strides = (1,1), padding = 'same', kernel_initializer = initializers.glorot_uniform(), kernel_regularizer=regularizers.l2(0.0005))(X)
    X = layers.BatchNormalization(axis = 3)(X)
    X = layers.Activation('relu')(X)

    #X = layers.Dropout(0.3) (X)

    X = layers.Conv2D(filters = 256, kernel_size = (1, 1), strides = (1,1), padding = 'same', kernel_initializer = initializers.glorot_uniform(), kernel_regularizer=regularizers.l2(0.0005))(X)
    X = layers.BatchNormalization(axis = 3)(X)
    X = layers.Activation('relu')(X)
    
    # Stage 3 up
    X = convolutional_block_inv(X, f=2, filters=[64, 128, 128], s=2)
    X = identity_block_inv(X, 2, [64, 128, 128])

    # Stage 2 up
    X = convolutional_block_inv(X, f = 4, filters = [32, 64, 64], s = 1)
    X = identity_block_inv(X, 4, [32, 64, 64])

    # Stage 1 up
    X = layers.Conv2DTranspose(32, (8, 8), strides = (2, 2), kernel_initializer = initializers.glorot_uniform(), padding='same', kernel_regularizer=regularizers.l2(0.0005))(X)
    X = layers.BatchNormalization(axis = 3)(X)
    X = layers.Activation('relu')(X)

    # long range shortcut
    X = layers.Concatenate(axis = 3)([X, inputs])

    # output-layer
    X = layers.Dense(1, activation='sigmoid', kernel_initializer = initializers.glorot_uniform()) (X)
    
    # Create model
    model = models.Model(inputs = inputs, outputs = X, name='CA_Unet')

    optimizer = optimizers.Adam(lr=0.0001)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()

    return model
