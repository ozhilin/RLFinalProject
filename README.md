Amir Ardalan Kalantari Dehagh - 260851609
Oleg Zhilin - 260581713

# COMP 767 Final Project - Policy Adjustment in Robotics

Note that when cloning this repository, the saved models are included so it might take a bit of time (the total size is slightly over 1GB)

## Instructions

- Create a virtual environment: `python -m venv venv`
- Activate it: `source venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt`
- Add custom environments as dependencies: `cd analysis/continuous_cartpole; pip install -e .`
- Finally, run one of the following files from the `analysis` directory:
  - `cartpole.py`: to train PILCO models (takes a long time)
  - `cartpole_from_file.py`: visualize learning of PILCO models in the `saved` directory
  - `train_transfer.py`: train a set of transfer models for each PILCO iteration in the `saved` directory
  - `path_analysis.py`: to reproduce path plots from the report (needs `saved` and `transfer-save` directory contents)
