# AI Chess Agent Using Supervised Imitation Learning and MCTS
This repository contains code for training an AI chess agent using supervised imitation learning and Monte Carlo tree search (MCTS). Weights of the pre-trained models are also saved in this repo as well since they are each <100MB. Below is a brief outline of the contents of this repo:
- `config`: This folder contains config files that specify the model architecture and training parameters for each model.
- `core`: This folder contains core code for this project including the definition of each model architecture in `torch_models.py` and search algorithms used to enhance the play performance of the models in `search_algos.py`
- `results`: This folder contains the results from training. The weights of the final models are stored in the checkpoints folder and an `eval_df.csv` is also included which tabulates all the evaluation games run with each model.
- `utils`: This module contains general code utility modules used throughout the project.
- `training.py`: This module is the main driver script for running supervised imitation learning for each model.
