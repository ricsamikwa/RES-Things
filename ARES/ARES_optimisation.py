
from distutils.log import error
from numpy import argmin
from numpy import asarray
from numpy import sum
import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import time
import numpy as np
import sys

sys.path.append('../')
import config
import utils
from Communicator import *
import multiprocessing
# from torchsummary import summary
import torchvision.models as models
# from torch.profiler import profile, record_function, ProfilerActivity




import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

np.random.seed(0)
torch.manual_seed(0)

class BenchClient(Communicator):
	def __init__(self, index, ip_address, datalen, model_name, split_layer):
		super(BenchClient, self).__init__(index, ip_address)
		self.datalen = datalen
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.model_name = model_name
		self.uninet = utils.get_model('Unit', self.model_name, config.model_len-1, self.device, config.model_cfg)
		# print(sys.getsizeof(self.uninet))
		# logger.info('Connecting to Server.')
		# self.sock.connect((server_addr,server_port))

	def initialize(self, split_layer, offload, first, LR):
		if offload or first:
			self.split_layer = split_layer[0]

			logger.debug('Building Model.')
			self.net = utils.get_model('Client', self.model_name, self.split_layer, self.device, config.model_cfg)
			self.server_net = utils.get_model('Server', self.model_name, self.split_layer, self.device, config.model_cfg)

			logger.debug(self.net)
			self.criterion = nn.CrossEntropyLoss()
		self.device_optimizer = optim.SGD(self.net.parameters(), lr=LR,
					  momentum=0.9)
		self.server_optimizer = optim.SGD(self.net.parameters(), lr=LR,
					  momentum=0.9)
		logger.debug('Done building the model..')

	def trace_handler(p):
		output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
		print(output)
		p.export_chrome_trace("/tmp/trace_" + str(p.step_num) + ".json")
		
	def train(self, trainloader):

		# Training start
		s_time_total = time.time()

		forward_time = 0	
		forward_end_time = 0
		server_forward_end_time = 0
		server_backward_end_time = 0
		device_backward_end_time = 0

		self.net.to(self.device)
		self.net.train()
		if self.split_layer == (config.model_len -1): # No offloading training
			for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(trainloader)):
				inputs, targets = inputs.to(self.device), targets.to(self.device)
				self.device_optimizer.zero_grad()
				# new random stuff
				forward_time = time.time()	
				outputs = self.net(inputs)				
				loss = self.criterion(outputs, targets)
				forward_end_time = time.time()

				loss.backward()
				self.device_optimizer.step()
				device_backward_end_time = time.time()

				break

			
		else: # Offloading training
			for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(trainloader)):
				inputs, targets = inputs.to(self.device), targets.to(self.device)
				
				# BenchClient forward
				forward_time = time.time()	
				self.device_optimizer.zero_grad()
				outputs = self.net(inputs)
				BenchClient_output = outputs.clone().detach().requires_grad_(True)
				forward_end_time = time.time()
				
				# print(sys.getsizeof(inputs))
				# print(sys.getsizeof(outputs))
				# print(sys.getsizeof(BenchClient_output))

				# server forward/backward
				self.server_optimizer.zero_grad()
				outputs_server = self.server_net(BenchClient_output)
				loss = self.criterion(outputs_server, targets)
				server_forward_end_time = time.time()
				
				loss.backward()
				self.server_optimizer.step()
				server_backward_end_time = time.time()

				#BenchClient backward
				BenchClient_grad = BenchClient_output.grad.clone().detach()
				outputs.backward(BenchClient_grad)
				self.device_optimizer.step()
				device_backward_end_time = time.time()
				# print("split layer: "+str(self.split_layer) + " forward device: "+ str(forward_end_time - forward_time)+" server forward: "+str(server_forward_end_time - forward_end_time))
				# print("server backward "+str(server_backward_end_time - server_forward_end_time)+ " device backward "+ str(device_backward_end_time - server_backward_end_time))

				break

		forward_device = forward_end_time - forward_time
		forward_server = server_forward_end_time - forward_end_time
		backward_server = server_backward_end_time - server_forward_end_time
		backward_device = device_backward_end_time - server_backward_end_time

		# print(self.split_layer)
		if self.split_layer == 6:
			forward_device = forward_end_time - forward_time
			backward_device = device_backward_end_time - forward_end_time
			forward_server = 0
			backward_server = 0


		return forward_device, forward_server, backward_server, backward_device
		
	def upload(self):
		msg = ['MSG_LOCAL_WEIGHTS_BenchClient_TO_SERVER', self.net.cpu().state_dict()]
		self.send_msg(self.sock, msg)

	def reinitialize(self, split_layers, offload, first, LR):
		self.initialize(split_layers, offload, first, LR)

# def local_layerwise_time():
#     forward_layerwise_latency = [0.020343, 0.033343, 0.023343, 0.012343, 0.03467367, 0.01111876, 0.00213267]
#     backward_layerwise_latency = [0.020343, 0.033343, 0.023343, 0.012343, 0.03467367, 0.01111876, 0.00213267]

#     return forward_layerwise_latency, backward_layerwise_latency
# def server_layerwise_time():
#     forward_layerwise_latency = [0.0020343, 0.0033343, 0.0023343, 0.0012343, 0.003467367, 0.001111876, 0.000213267]
#     backward_layerwise_latency = [0.0020343, 0.0033343, 0.0023343, 0.0012343, 0.003467367, 0.001111876, 0.000213267]
#     return forward_layerwise_latency, backward_layerwise_latency

# def error_calculation():
#     error_calc_time = 0.0001
#     return error_calc_time

	def transmission_layerwise_time(self, network_throughput):
		layerwise_data = [534309, 653343, 912343, 534309, 534309, 534309]
		# layerwise_latency = [element * (1/network_throughput) for element in layerwise_data]
		layerwise_latency = [0.3262598514556885, 0.5824382305145264, 0.13750505447387695, 0.13750505447387695, 0.6942946910858154, 0.10664844512939453]

		# 40 Mbit/s 
		# layerwise_latency = [0.5517283058166504, 0.13045337677001953, 0.1769771146774292, 0.18822888851165773, 0.001455254554748535, 0.0]
		layerwise_latency_backpropagation = [1.569196753501892, 0.6807714366912841, 0.3989624071121216, 0.3490042543411255, 0.09093882560729981, 0.0]


		# print(layerwise_latency)
		return layerwise_latency, layerwise_latency_backpropagation

	def measure_power(self):
		computation_power = 5400
		transmission_power = 3100

		return computation_power, transmission_power

	def training_time_energy(self):
		
		offload = True
		first = True # First initializaiton control
		# first = True # First initializaiton control

		self.initialize([6], offload, first, config.LR)
		# first = False 
		first = True 
		# this has problems on mac
		cpu_count = multiprocessing.cpu_count()
		trainloader, classes= utils.get_local_dataloader(1, 0)

		device_forward_splitwise_latency = [0,0,0,0,0,0]
		server_forward_splitwise_latency = [0,0,0,0,0,0]
		server_backward_splitwise_latency = [0,0,0,0,0,0]
		device_backward_splitwise_latency = [0,0,0,0,0,0]

		# config.split_layer = 6
		for r in range(config.model_len - 1, 0, -1):
			# config.split_layer = r
			if r < config.model_len - 1:
				self.reinitialize([r], offload, first, config.LR)

			# print(config.split_layer)
			forward_device, forward_server, backward_server, backward_device = self.train(trainloader)
			device_forward_splitwise_latency[r -1] = forward_device
			server_forward_splitwise_latency[r -1] = forward_server
			server_backward_splitwise_latency[r -1] = backward_server
			device_backward_splitwise_latency[r -1] = backward_device

			if r > 49:
				LR = config.LR * 0.1
			
			
		
		print(str(device_forward_splitwise_latency)+"\n"+str(server_forward_splitwise_latency)+"\n"+ str(device_backward_splitwise_latency)+ "\n"+str(server_backward_splitwise_latency))

		# # nano 8 - MAXN
		device_forward_splitwise_latency_temp = [0.0023119449615478516, 0.003648519515991211, 0.003835916519165039, 0.006833791732788086, 0.021893978118896484, 9.588886499404907]
		device_backward_splitwise_latency_temp = [0.003223419189453125, 0.030652284622192383, 0.00543975830078125, 0.0074024200439453125, 0.007901191711425781, 1.5804917812347412]

		# pi B 
		# device_forward_splitwise_latency_temp = [0.3574063777923584, 0.8774583339691162, 1.0404078960418701, 1.163121223449707, 1.2735788822174072, 1.314896583557129]
		# device_backward_splitwise_latency_temp = [0.536374568939209, 1.4896900653839111, 1.4612138271331787, 1.8985178470611572, 1.9876246452331543, 1.7998700141906738]

		
		trans_layerwise_time, layerwise_latency_backpropagation = self.transmission_layerwise_time(2000000)
		
		device_training_computation_time_array = []
		server_training_computation_time_array = []
		total_training_time_array = []

		device_training_computation_time_array = [(device_forward_splitwise_latency_temp[i] + device_backward_splitwise_latency_temp[i]) for i in range(len(device_forward_splitwise_latency))]
		server_training_computation_time_array = [(server_forward_splitwise_latency[i] + server_backward_splitwise_latency[i]) for i in range(len(device_forward_splitwise_latency))]
		total_training_time_array = [(device_training_computation_time_array[p] + server_training_computation_time_array[p] + trans_layerwise_time[p]) for p in range(len(device_forward_splitwise_latency))]

		computation_power, transmission_power = self.measure_power()

		splitwise_computation_energy = [element * computation_power for element in device_training_computation_time_array]
		layerwise_transmission_energy = [element * 2 * transmission_power for element in trans_layerwise_time] 

		total_energy_per_iter_array = []

		total_energy_per_iter_array = [(splitwise_computation_energy[i] + layerwise_transmission_energy[i]) for i in range(len(splitwise_computation_energy))]

		return total_training_time_array, total_energy_per_iter_array

	def ARES_optimiser(self, alpha):
		print("alpha: " + str(alpha))
		total_training_time_array, total_energy_per_iter_array = self.training_time_energy()
		
		#normalisation
		norm = np.linalg.norm(total_training_time_array)
		normal_training_time_array = total_training_time_array/norm
		
		norm2 = np.linalg.norm(total_energy_per_iter_array)
		normal_energy_per_iter_array = total_energy_per_iter_array/norm2
		
		print("==============================")
		print(normal_training_time_array)
		print(normal_energy_per_iter_array)
		print("==============================")
		print("training time argmin: "+ str(argmin(normal_training_time_array) + 1))
		print("energy consump argmin: "+ str(argmin(normal_energy_per_iter_array) +1))
		print("==============================")

		#scaling
		scaled_normal_training_time_array = [element * alpha for element in normal_training_time_array]
		
		scaled_normal_energy_per_iter_array = [element * (1 - alpha) for element in normal_energy_per_iter_array]
		
		optimisation_array = np.add(scaled_normal_training_time_array, scaled_normal_energy_per_iter_array)  

		# print(optimisation_array)
		
		# axis 0 for now
		result = argmin(optimisation_array, axis=0)
		
		return result


logger.info('Preparing Device')
benchClient = BenchClient(1, '192.168.1.100', 50000, 'VGG5', 6)

s_time_rebuild = time.time()
offloading_strategy = benchClient.ARES_optimiser(0.4) + 1
e_time_rebuild = time.time()
print("Current offloading strategy: "+ str(offloading_strategy))
print(('Optimisation time: ' + str(e_time_rebuild - s_time_rebuild)))