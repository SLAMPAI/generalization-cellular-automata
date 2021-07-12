# Generalization over different cellular automata rules learned by a deep feed-forward neural network

*by Marcel Aach, Jens Henrik Goebbert, Jenia Jitsev* https://arxiv.org/abs/2103.14886

## Introduction 

In this repository we provide the code for reproducing the experiments performend in the Paper "Generalization over different cellular automata rules learned by a deep feed-forward neural network". 

## Organization

```
------------
	├── README.md		   					 <- README doc for reproducing experiments
	├── data_generator.py			   <- data generator and helper function to generate the different CA trajectories
	├── ca_unet.py						   <- the encoder-decoder neural network model
	├── training.py 					   <- the script for running the training (and testing) the network 
	├── params
			├── params_simple			   <- the parameters for the simple generalization case
			├── params_level_1		   <- the parameters for the level 1 generalization case
			├── params_level_2		   <- the parameters for the level 2 generalization case
			├── params_level_3_extra <- the parameters for the level 3 extrapolation generalization case
			├── params_level_3_inter <- the parameters for the level 3 interpolation generalization case
  ├── juwels_run.sh						 <- batch script to run the code on the JUWELS supercomputer at FZ Juelich
------------
```

## Installation

For running the scripts only TensorFlow 2.3.1 and Python 3.8.5 are required.

## How to run 

Run the training.py python file with one of the following flags:

- simple: for the simple generalization using only 3x3 neighborhood in training and test set
- 1: for the level 1 generalization using 3x3, 5x5 and 7x7 neighborhood in training and test set
- 2: for the level 2 generalization using 3x3, 5x5 and 7x7 neighborhood in training but different rules in test set
- extra: for the level 3 generalization using 3x3, 5x5 and 7x7 neighborhood in training but 9x9 in test set
- inter: for the level 3 generalization using 3x3, 5x5 and 9x9 neighborhood in training but 7x7 in test set

Other hyperparameters (like number of epochs or batch size) can be set in the corresponding parameter file in the params folder. 



## Citation

If you find this work helpful, please cite our paper:

```
​```
@misc{aach2021generalization,
      title={Generalization over different cellular automata rules learned by a deep feed-forward neural network}, 
      author={Marcel Aach and Jens Henrik Goebbert and Jenia Jitsev},
      year={2021},
      eprint={2103.14886},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
​```
```



