
# numpy implementation of argmax
from distutils.log import error
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
    layerwise_data = [3343, 3343, 12343, 467367, 1876, 67]
    layerwise_latency = [element * (1/network_throughput) for element in layerwise_data]

    return layerwise_latency

def error_calculation():
    error_calc_time = 0.0001
    return error_calc_time

def training_time():
    device_forward_layerwise_latency, device_backward_layerwise_latency = local_layerwise_time()
    server_forward_layerwise_latency, server_backward_layerwise_latency = server_layerwise_time()
    trans_layerwise_time = transmission_layerwise_time()
    error_calculation = error_calculation()
    
    training_computation_time_array = []
    total_training_time_array = []

    for i in len(device_forward_layerwise_latency):
        training_computation_time_array[i] = sum(device_forward_layerwise_latency[0:i]) + sum(server_forward_layerwise_latency[i:]) + sum(server_backward_layerwise_latency[:i]) + sum(device_backward_layerwise_latency[:0]) 
    
    for p in len(device_forward_layerwise_latency):
        total_training_time_array[p] = training_computation_time_array[p] + trans_layerwise_time[p]

    return training_computation_time_array, total_training_time_array

def measure_power():
    computation_power = 5000
    transmission_power = 4000

    return computation_power, transmission_power

def energy_consumption():
    
    computation_power, transmission_power = measure_power()
    training_computation_time_array, total_training_time_array = training_time()
    trans_layerwise_time = transmission_layerwise_time()

    layerwise_computation_energy = [element * computation_power for element in training_computation_time_array]
    layer


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