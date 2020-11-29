# MR. NODE
MR. NODE (Multiple predictoR Neural ODE) is a deep learning method that models the infection rate of Black sigatoka via ordinary differential equations, and which can infer the infection risk variable at an arbitrary point on the timeline.

This is the submission of the University of Toronto team to [ProjectX 2020](https://www.projectx2020.com/), a machine learning research competition.


# Usage

## Installation
1. Install [Poetry](https://python-poetry.org/).
2. Clone this repository and `cd` into its directory.
3. Install the project and run the training script in the right environment.
```shell
$ poetry install
$ poetry shell
$ python train.py
```

## Training
You may use command-line arguments.

# Results
![Sample extrapolation result](/images/CR_100_2.png)