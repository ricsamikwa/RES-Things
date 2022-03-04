
from distutils.log import error
from numpy import argmin
from numpy import asarray
from numpy import sum
import numpy as np
import time
 


def local_layerwise_time():
    forward_layerwise_latency = [0.020343, 0.033343, 0.023343, 0.012343, 0.03467367, 0.01111876, 0.00213267]
    backward_layerwise_latency = [0.020343, 0.033343, 0.023343, 0.012343, 0.03467367, 0.01111876, 0.00213267]

    return forward_layerwise_latency, backward_layerwise_latency
def server_layerwise_time():
    forward_layerwise_latency = [0.0020343, 0.0033343, 0.0023343, 0.0012343, 0.003467367, 0.001111876, 0.000213267]
    backward_layerwise_latency = [0.0020343, 0.0033343, 0.0023343, 0.0012343, 0.003467367, 0.001111876, 0.000213267]
    return forward_layerwise_latency, backward_layerwise_latency
    
def transmission_layerwise_time(network_throughput):
    layerwise_data = [5343009, 653343, 912343, 47367, 1876, 6789, 7386]
    layerwise_latency = [element * (1/network_throughput) for element in layerwise_data]

    return layerwise_latency

def error_calculation():
    error_calc_time = 0.0001
    return error_calc_time

def training_time():
    device_forward_layerwise_latency, device_backward_layerwise_latency = local_layerwise_time()
    server_forward_layerwise_latency, server_backward_layerwise_latency = server_layerwise_time()
    trans_layerwise_time = transmission_layerwise_time(2000000)
    error_calculation_time = error_calculation()
    
    training_computation_time_array = []
    total_training_time_array = []

    training_computation_time_array = [(sum(device_forward_layerwise_latency[0:i]) + sum(server_forward_layerwise_latency[i:]) + sum(server_backward_layerwise_latency[:i]) + sum(device_backward_layerwise_latency[:0]) + error_calculation_time) for i in range(len(device_forward_layerwise_latency))]
    # for i in range(len(device_forward_layerwise_latency)):
    #     training_computation_time_array = (sum(device_forward_layerwise_latency[0:i]) + sum(server_forward_layerwise_latency[i:]) + sum(server_backward_layerwise_latency[:i]) + sum(device_backward_layerwise_latency[:0]) + error_calculation_time)
    total_training_time_array = [(training_computation_time_array[p] + trans_layerwise_time[p]) for p in range(len(device_forward_layerwise_latency))]

    return training_computation_time_array, total_training_time_array

def measure_power():
    computation_power = 5400
    transmission_power = 3100

    return computation_power, transmission_power

def energy_consumption():
    
    computation_power, transmission_power = measure_power()
    training_computation_time_array, total_training_time_array = training_time()
    trans_layerwise_time = transmission_layerwise_time(2000000)

    layerwise_computation_energy = [element * computation_power for element in training_computation_time_array]
    layerwise_transmission_energy = [element * transmission_power for element in trans_layerwise_time]

    total_energy_per_iter_array = []

    total_energy_per_iter_array = [(2 * sum(layerwise_computation_energy[0:i]) + layerwise_transmission_energy[i]) for i in range(len(layerwise_computation_energy))]
    
    return total_energy_per_iter_array

def ARES_optimiser(alpha):
    print("alpha: " + str(alpha))
    training_computation_time_array, total_training_time_array = training_time()
    total_energy_per_iter_array = energy_consumption()
    
    #normalisation
    norm = np.linalg.norm(total_training_time_array)
    normal_training_time_array = total_training_time_array/norm
    
    norm2 = np.linalg.norm(total_energy_per_iter_array)
    normal_energy_per_iter_array = total_energy_per_iter_array/norm2
    
    print("==============================")
    print(normal_training_time_array)
    print(normal_energy_per_iter_array)
    print("==============================")
    print("training time argmin: "+ str(argmin(normal_training_time_array)))
    print("energy consump argmin: "+ str(argmin(normal_energy_per_iter_array)))
    print("==============================")

    #scaling
    scaled_normal_training_time_array = [element * alpha for element in normal_training_time_array]
    
    scaled_normal_energy_per_iter_array = [element * (1 - alpha) for element in normal_energy_per_iter_array]
    
    optimisation_array = np.add(scaled_normal_training_time_array, scaled_normal_energy_per_iter_array)  

    # print(optimisation_array)
    
    # axis 0 for now
    result = argmin(optimisation_array, axis=0)
    
    return result
s_time_rebuild = time.time()
offloading_strategy = ARES_optimiser(0.4)
e_time_rebuild = time.time()
print("Current offloading strategy: "+ str(offloading_strategy))
print(('Optimisation time: ' + str(e_time_rebuild - s_time_rebuild)))