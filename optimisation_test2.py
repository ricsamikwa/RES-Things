
# numpy implementation of argmax
from numpy import argmax
from numpy import asarray
from numpy import sum


def local_layerwise_time():
    forward_layerwise_latency = [0.020343, 0.033343, 0.023343, 0.012343, 0.03467367, 0.01111876, 0.00213267]
    backward_layerwise_latency = [0.020343, 0.033343, 0.023343, 0.012343, 0.03467367, 0.01111876, 0.00213267]

    return forward_layerwise_latency, backward_layerwise_latency
def server_layerwise_time():
    forward_layerwise_latency = [0.0020343, 0.0033343, 0.0023343, 0.0012343, 0.003467367, 0.001111876, 0.000213267]
    backward_layerwise_latency = [0.020343, 0.033343, 0.023343, 0.012343, 0.03467367, 0.01111876, 0.00213267]
    return forward_layerwise_latency, backward_layerwise_latency
    
def transmission_layerwise_time(network_throughput):
    layerwise_data = [0.0033343, 0.0023343, 0.0012343, 0.003467367, 0.001111876, 0.000213267]
    layerwise_latency = 1
    return layerwise_data

def training_time(layer):
    device_layerwise_time = local_layerwise_time()
    se_layerwise_time = server_layerwise_time()
    trans_layerwise_time = transmission_layerwise_time()
    
    training_time_array = []

    for i in len(device_layerwise_time):
        training_time_array[i] = sum(device_layerwise_time[0:i]) + sum(se_layerwise_time[i:]) + trans_layerwise_time[i]
    
    return training_time_array
def power_measured():
    layerwise_power = [0.0033343, 0.0023343, 0.0012343, 0.003467367, 0.001111876, 0.000213267]
    return layerwise_power
def energy_consumption():
    ghsdg
def combination():
    ghsdg
def combination_with_alpha():
    ghsdg
# define vector
probs = asarray([[0.4, 0.5, 0.1], [0.0, 0.0, 1.0], [0.9, 0.0, 0.1], [0.3, 0.3, 0.4]])
print(probs.shape)
# get argmax
result = argmax(probs, axis=1)
print(result)