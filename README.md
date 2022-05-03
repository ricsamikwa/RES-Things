# Adaptive REsource-aware Split learning (ARES) for Internet of Things
### A scheme for efficient decentralised training in IoT systems. 
ARES accelerates local training in resource-constrained devices and minimizes the effect of stragglers on the training through device-targeted split points while accounting for time-varying network throughput and computing resources. ARES takes into account application constraints (energy sensitivity policy) to mitigate training optimization tradeoffs in terms of energy consumption and training time.

## Requirements

### Setting up the environment
Python 3 with Pytorch version 1.4 and torchvision 0.5. IoT device (NVIDIA Jetson Nano and Raspberry Pi). 
Some pre-built PyTorch and torchvision pip wheel:

- Jetson: https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-8-0-now-available/72048
- Pyotrch: https://discuss.pytorch.org/t/pytorch-1-3-wheels-for-raspberry-pi-python-3-7/58580

#### Packages
* socket
* struct
* pickle
* numpy
* threading
* time
* matplotlib
* torch
* torchvision
* sys
* json

#### Dataset
CIFAR10 datasets can be downloaded manually and put into the `datasets/CIFAR10` folder. 
- CIFAR10: https://www.cs.toronto.edu/~kriz/cifar.html
- Altenatively, set the download parameter to True (on initial run) in torchvision.datasets.CIFAR10 in functions.py
