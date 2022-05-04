# Adaptive REsource-aware Split learning (ARES) for Internet of Things
### A scheme for efficient model training in IoT systems. 

ARES accelerates training in resource-constrained devices and minimizes the effect of stragglers on the training through device-targeted split points while accounting for time-varying network throughput and computing resources. ARES takes into account application constraints (using energy sensitivity policy) to mitigate training optimization tradeoffs in terms of energy consumption and training time.

## Requirements

### Setting up the environment

Python 3.7+ with Pytorch and torchvision 0.10 for IoT devices (NVIDIA Jetson Nano and Raspberry Pi) and Server. 
Some pre-built PyTorch and torchvision pip wheel:

- Jetson Nano: https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-8-0-now-available/72048
- Pytorch (Pi): https://discuss.pytorch.org/t/pytorch-1-3-wheels-for-raspberry-pi-python-3-7/58580

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
* csv

#### Dataset

CIFAR10 datasets can be downloaded manually and put into the `datasets/CIFAR10` folder. 
- CIFAR10: https://www.cs.toronto.edu/~kriz/cifar.html
- Altenatively, set the download parameter to True (on initial run) in torchvision.datasets.CIFAR10 in functions.py

### Configurations 

To set the edge server and IoT devices configurations, edit the file configurations.py with the corresponding IP addresses, hostnames and port numbers for all devices and server. 
The default port number is set to 43000 for all devices. The configuration file is shared on all devices and server.

## Usage

### Running training using ARES 

Launch ARES training on the server in `ARES_training` folder:
```
python3 ARES_Server.py --split True 
```
Launch ARES training on the IoT devices:
```
python3 ARES_Device.py --split True #requires sudo on Jetson Nano
```

The --split argument specifies an option to process all layers of the neural network model on the devices:
```
python3 ARES_Server.py --split False 
python3 ARES_Device.py --split False 
```

Run test ARES optimisation in `ARESopt` folder:
```
python3 ARES_optimiser.py
```
This requires all packages to installed.

### Switching power modes on Jetson Nano

The nvpmodel.conf file in `Nano_power_modes` folder resource configurations (power modes) for the Jetson Nano and can be manipulated to create custom ones. Update the change in `/etc/nvpmodel.conf` file.

To check the current power mode run:
```
sudo /usr/sbin/nvpmodel -q
```
To switch to another mode run:
```
sudo /usr/sbin/nvpmodel -m [MODE NUMBER] #the MODE NUMBER corresponds to the position of the target mode starting from 0
```
Note: the jtop interface for monitoring resources on the Jetson Nano sometimes does not run properly for custom power modes.

power monitoring in `I2C` folder:
```
/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/in_power0_input
```

### Network traffic shapping

Modify the traffic_shapper.sh file in `Network_traffic` folder with the corresponding IP addresses and bandwidth limits. The qdiscs can be set for different network configurations accordingly.

To modify network traffic run:  
```
sudo sh ./traffic_shapper.sh
```
To clear the qdisc on the device run:
To modify network traffic run:  
```
sudo sh ./clean_traffic_shapper.sh
```
