# Identifying Latent State-Transition Processes for Individualized Reinforcement Learning
## Overview
In recent years, the application of reinforcement learning (RL) involving interactions with individuals has seen significant growth. These interactions, influenced by individual-specific factors ranging from personal preferences to physiological differences, can causally affect state transitions, such as the health conditions in healthcare or learning progress in education. Consequently, different individuals may exhibit different state-transition processes. Understanding these individualized state-transition processes is crucial for optimizing individualized policies. In practice, however, identifying these state-transition processes is challenging, especially since individual-specific factors often remain latent. In this paper, we establish the identifiability of these latent factors and present a practical method that effectively learns these processes from observed state-action trajectories. Our experiments on various datasets show that our method can effectively identify the latent state-transition processes and help learn individualized RL policies.
## Requirements
To install it, create a conda environment with Python>=3.6 and follow the instructions below. Note that the current implementation requires GPUs.
```ruby
conda create -n ivae python=3.6.13
cd ivae
pip install -e .
```
