# Bayesian Action Decoder
PyTorch version of the proof-of-principle [implementation](https://bit.ly/2P3YOyd) of the [Bayesian Action Decoder](https://arxiv.org/abs/1811.01458) for a two-step matrix game.

## Files
- **bad.py** Trains the three different agents (Vanilla PG, Bayesian Action Decoder, BAD with counterfactual actions). 
Adjust the number of runs and episodes used by toggling `debug`. Switch to only viewing results from a file by setting `skip_training=True`.
- **no_baseline.py** Trains the agents without using baselines. 
- **logs** Contains results from training the agents averaged over 10 runs, using 15 000 episodes and a batch size of 32.