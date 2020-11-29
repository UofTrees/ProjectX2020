# MR. NODE
MR. NODE (Multiple predictoR Neural ODE) is a deep learning method that models the infection rate of Black sigatoka via ordinary differential equations, and which can infer the infection risk variable at an arbitrary point on the timeline.

This is the submission of the University of Toronto team to [ProjectX 2020](https://www.projectx2020.com/), a machine learning research competition.

```
.
├── baseline          # All files related to baseline models
├── mr_node           # Data structures for MR. NODE
├── train.py          # Training script for MR. NODE
├── test.py           # Testing script for MR. NODE 
└── data              # Time series data for 7 regions. We use only 2 of them: Costa Rica and India.
```

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

## MR. NODE
### Training
You may use the following command to train the model. The results can be found in `/results`.
```shell
$ python train.py --region=cr --lr=3e-4 --encoder_fc_dims 8 16 8 --hidden_dims=4 --odefunc_fc_dims 64 64 --decoder_fc_dims 64 64 --window_length=128 --num_epochs=1 --rtol=1e-4 --atol=1e-6
```
Keyword arguments:
- `region`: Whether to train using data from Costa Rica (`cr`), India (`in`), or both (`crin`). Any other argument will raise an error. Default: `cr`
- `lr`: learning rate. Default: `3e-4`
- `encoder_fc_dims`: Fully-connected layers in the encoder. Default: `8 16 8`
- `hidden_dims`: Dimensions of latent space. Default: `4`
- `odefunc_fc_dims`: Fully-connected layers in the dynamics function. Default: `64 64`
- `decoder_fc_dims`: Fully-connected layers in the decoder. Default: `8 16 8`
- `window_length`: Window length for time steps. Default: `128`
- `num_epochs`: Number of training epochs. Default: `1`
- `rtol`: Relative tolerance for Neural ODE. Default: `1e-4`
- `atol`: Absolute tolerance for Neural ODE. Default: `1e-4`

### Testing
Training a model with a set of arguments will generate a `.pt` file in `/results/models` uniquely identified by a "`job_id`" created based on the training arguments. You may use this `job_id` to specify which model to test.

```shell
$ python test.py --region=cr --job_id='cr_lr3.0e-04_enc[8, 16, 8]_hidden4_ode[64, 64]_dec[64, 64]_window128_epochs1_rtol0.0001_atol1e-06' --plot_indiv=False --num_to_keep=100
```
Keyword arguments:
- `region`: Whether to test using data from Costa Rica (`cr`), India (`in`), or both (`crin`). Any other argument will raise an error. Default: `cr`
- `job_id`: Number of training epochs. Default: `cr_lr3.0e-04_enc[8, 16, 8]_hidden4_ode[64, 64]_dec[64, 64]_window128_epochs1_rtol0.0001_atol1e-06`
- `plot_indiv`: Whether or not to generate individual plots in `results/plots`. If not, all the plots will be created on a single image file. Default: `False`
- `num_to_keep`: Number of time steps to use to create the initial latent state. Default: `100`

## Baselines

# Results
![Sample extrapolation result](/images/CR_100_2.png)