# MR. NODE
MR. NODE (Multiple predictoR Neural ODE) is a deep learning method that models the infection rate of black Sigatoka via ordinary differential equations, and which can infer the infection risk variable at an arbitrary point on the timeline.

This is the submission of the University of Toronto team to [ProjectX 2020](https://www.projectx2020.com/), an international machine learning research competition.

```
.
├── baseline          # All files related to baseline models
├── mr_node           # Data structures for MR. NODE
├── train.py          # Training script for MR. NODE
├── test.py           # Testing script for MR. NODE 
└── data              # Time series data for Costa Rica and India.
```

![Sample extrapolation result](/images/result2.png)

# Data
We have collected microclimatic data from India and Costa Rica, two regions of the world known for having vast banana plantations and synthesized the corresponding infection risk variable via a probabilistic survival process inspired by [2].

| Region             | Latitude | Longitude |
|--------------------|----------|-----------|
| Costa Rica         | 10.39    | -83.812   |
| Maharashtra, India | 18.8143  | 73.125    |

# Installation
1. Install [Poetry](https://python-poetry.org/).
2. Clone this repository and `cd` into its directory.
3. Install the project and run the training script in the right environment.
```shell
$ poetry install
$ poetry shell
$ python train.py
```

# MR. NODE
## Training
You may use the following command to train the model. The results can be found in `/results`.
```shell
$ python train.py --region=cr --lr=3e-4 --encoder_fc_dims 8 16 8 --hidden_dims=4 --odefunc_fc_dims 64 64 --decoder_fc_dims 64 64 --window_length=128 --num_epochs=1 --rtol=1e-4 --atol=1e-6
```
Keyword arguments:
- `region`: Whether to train using data from Costa Rica (`cr`), India (`in`), or both (`crin`). Default: `cr`
- `lr`: learning rate. Default: `3e-4`
- `encoder_fc_dims`: Fully-connected layers in the encoder. Default: `8 16 8`
- `hidden_dims`: Dimensions of latent space. Default: `4`
- `odefunc_fc_dims`: Fully-connected layers in the dynamics function. Default: `64 64`
- `decoder_fc_dims`: Fully-connected layers in the decoder. Default: `8 16 8`
- `window_length`: Window length for time steps. Default: `128`
- `num_epochs`: Number of training epochs. Default: `1`
- `rtol`: Relative tolerance for Neural ODE. Default: `1e-4`
- `atol`: Absolute tolerance for Neural ODE. Default: `1e-4`

## Testing
Training a model with a set of arguments will generate a `.pt` file in `/results/models` uniquely identified by a `job_id` created based on the training arguments. You may use this `job_id` to specify which model to test.

```shell
$ python test.py --region=cr --job_id='cr_lr3.0e-04_enc[8, 16, 8]_hidden4_ode[64, 64]_dec[64, 64]_window128_epochs1_rtol0.0001_atol1e-06' --plot_indiv=False --num_to_keep=100
```
Keyword arguments:
- `region`: Whether to test using data from Costa Rica (`cr`), India (`in`), or both (`crin`). Default: `cr`
- `job_id`: Job id of the model to test. Default: `cr_lr3.0e-04_enc[8, 16, 8]_hidden4_ode[64, 64]_dec[64, 64]_window128_epochs1_rtol0.0001_atol1e-06`
- `plot_indiv`: Whether or not to generate individual plots in `results/plots`. If not, all the plots will be created on a single image file. Default: `False`
- `num_to_keep`: Number of time steps to use to create the initial latent state. This must be a positive integer no greater than 100.  Default: `100`

# Baseline RNN and LSTM
## Training
You may use the following command to train the baseline RNN or LSTM model. The results can be found in `/baseline/baseline_results`.
```shell
$ cd baseline
$ python train_baseline.py --region=cr --lr=0.001 --batch_size=256 --seq_len=100 --num_epochs=1 --n_hidden=20 --model_name=lstm
```
Keyword arguments:
- `region`: Whether to train using data from Costa Rica (`cr`), India (`in`), or both (`crin`). Default: `cr`
- `lr`: learning rate. Default: `0.001`
- `batch_size`: batch size. Default: `256`
- `seq_len`: Number of ground-truth points to use when extrapolating. This must be a positive integer no greater than 100.  Default: `100`
- `num_epochs`: Number of training epochs. Default: `1`
- `n_hidden`: Number of hidden units in the RNN/LSTM. Default: `20`
- `model_name`: Can be `lstm` or `rnn`. Default: `lstm`

## Testing
Training a model with a set of arguments will generate a `.pt` file in `/baseline/baseline_results/models` uniquely identified by a `job_id` created based on the training arguments. You may use this `job_id` to specify which model to test.

```shell
$ cd baseline
$ python test_baseline.py --region=cr --job_id='cr_lstm_lr1.0e-03_batch256_seq100_epochs1_hidden20'
```
Keyword arguments:
- `region`: Whether to test using data from Costa Rica (`cr`), India (`in`), or both (`crin`). Default: `cr`
- `job_id`: Job id of the model to test. Default: `cr_lstm_lr1.0e-03_batch256_seq100_epochs1_hidden20`

# References
[1] Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, and David Duvenaud. Neural Ordinary Differential Equations. 2018. https://arxiv.org/abs/1806.07366. <br/>
[2] Daniel P. Bebber.  Climate change effects on Black Sigatoka disease of banana. May 2019. https://doi.org/10.1098/rstb.2018.0269.
