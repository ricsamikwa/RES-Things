###################################

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
import configurations
import functions
from Wireless import *
import multiprocessing
# from torchsummary import summary
import torchvision.models as models
# from torch.profiler import profile, record_function, ProfilerActivity




import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

np.random.seed(0)
torch.manual_seed(0)

class BenchClient(Wireless):
	def __init__(self, index, ip_address, datalen, model_name, split_layer):
		super(BenchClient, self).__init__(index, ip_address)
		self.datalen = datalen
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.model_name = model_name
		self.uninet = functions.get_model('Unit', self.model_name, configurations.model_len-1, self.device, configurations.model_cfg)
		# print(sys.getsizeof(self.uninet))
		# logger.info('Connecting to Server.')
		# self.sock.connect((server_addr,server_port))

	def initialize(self, split_layer, split, first, LR):
		if split or first:
			self.split_layer = split_layer[0]

			self.net = functions.get_model('Client', self.model_name, self.split_layer, self.device, configurations.model_cfg)
			self.server_net = functions.get_model('Server', self.model_name, self.split_layer, self.device, configurations.model_cfg)

			logger.debug(self.net)
			self.criterion = nn.CrossEntropyLoss()
		self.device_optimizer = optim.SGD(self.net.parameters(), lr=LR,
					  momentum=0.9)
		self.server_optimizer = optim.SGD(self.net.parameters(), lr=LR,
					  momentum=0.9)
		logger.debug('Model created..')

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
		if self.split_layer == (configurations.model_len -1): 
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

			
		else: # spliting 
			for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(trainloader)):
				inputs, targets = inputs.to(self.device), targets.to(self.device)
				
				# BenchClient forward
				forward_time = time.time()	
				self.device_optimizer.zero_grad()
				outputs = self.net(inputs)
				BenchClient_output = outputs.clone().detach().requires_grad_(True)
				forward_end_time = time.time()
				
				# print(inputs.shape)
				# print(BenchClient_output.shape)


				# server forward/backward
				self.server_optimizer.zero_grad()
				outputs_server = self.server_net(BenchClient_output)
				# print(outputs_server.shape)

				loss = self.criterion(outputs_server, targets)
				# print(loss.shape)

				server_forward_end_time = time.time()
				# print("feature maps: "+str(sys.getsizeof(BenchClient_output)))

				loss.backward()
				self.server_optimizer.step()
				server_backward_end_time = time.time()

				#BenchClient backward
				BenchClient_grad = BenchClient_output.grad.clone().detach()
				# print(BenchClient_grad.shape)
				# print("gradients: "+str(sys.getsizeof(BenchClient_grad)))

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

	def reinitialize(self, split_layers, split, first, LR):
		self.initialize(split_layers, split, first, LR)

# def error_calculation():
#     error_calc_time = 0.0001
#     return error_calc_time

	def transmission_layerwise_time(self, network_throughput):

			# 'VGG' : [('C', 3, 32, 3, 32*32*32, 32*32*32*3*3*3), ('M', 32, 32, 2, 32*16*16, 0), 
			# ('C', 32, 64, 3, 64*16*16, 64*16*16*3*3*32), ('M', 64, 64, 2, 64*8*8, 0), 
			# ('C', 64, 64, 3, 64*8*8, 64*8*8*3*3*64), 
			# ('D', 8*8*64, 128, 1, 64, 128*8*8*64), 
			# ('C', 64, 64, 3, 64*8*8, 64*8*8*3*3*64), 
			# ('D', 8*8*64, 128, 1, 64, 128*8*8*64), 
			# ('D', 128, 10, 1, 10, 128*10)]

		# now we multiply everything by 8 * 100 - "why?" Answer: 1 byte is 8 bits AND 100 is batch size 
		
		forward_layerwise_data = [32*32*32*8*100, 32*16*16*8*100, 64*16*16*8*100, 64*8*8*8*100, 64*8*8*8*100, 128*8*100, 10*8*100]
		backward_layerwise_data = [32*32*32*8*100, 32*16*16*8*100, 64*16*16*8*100, 64*8*8*8*100, 64*8*8*8*100, 128*8*100, 10*8*100]

		forward_layerwise_latency = [element * (1/(network_throughput * 1000000)) for element in forward_layerwise_data]

		# print(forward_layerwise_latency)
		# const_forward_layerwise_latency = [0.3262598514556885, 0.5824382305145264, 0.13750505447387695, 0.13750505447387695, 0.6942946910858154, 0.10664844512939453]
		# print(const_forward_layerwise_latency)
		
		backward_layerwise_latency = [element * (1/(network_throughput * 1000000)) for element in backward_layerwise_data]
		# const_backward_layerwise_latency = [1.569196753501892, 0.6807714366912841, 0.3989624071121216, 0.3490042543411255, 0.09093882560729981, 0.0]
		print("transmission latency "+str(backward_layerwise_latency))
		# print(layerwise_latency)
		return forward_layerwise_latency, backward_layerwise_latency

	def measure_power(self):

		# temp
		# MAXN comp: 7253.297700323392
		# MAXN trans: 2319.590057210495
		# MAXN rec: 2260.3782324677427
		# 5W comp: 4204.6259168704155
		# 5W trans: 1917.0524958555905
		# 5W rec: 1912.4423741971912
		# CUSTOM comp: 2396.2112226277372
		# CUSTOM trans: 1754.1256388811448
		# CUSTOM rec: 1755.296656187482

		
		computation_power = 7253
		transmission_power = 2319
		receiving_power = 2260

		# computation_power = 3800
		# transmission_power = 1100
		# receiving_power = 800

		return computation_power, transmission_power, receiving_power

	def training_time_energy(self, throughput):
		
		split = True
		first = True 

		self.initialize([6], split, first, configurations.LR)
		# first = False 
		first = True 
		# this has problems on mac
		cpu_count = multiprocessing.cpu_count()
		trainloader, classes= functions.get_local_dataloader(1, cpu_count)

		device_forward_splitwise_latency = [0,0,0,0,0,0,0]
		server_forward_splitwise_latency = [0,0,0,0,0,0,0]
		server_backward_splitwise_latency = [0,0,0,0,0,0,0]
		device_backward_splitwise_latency = [0,0,0,0,0,0,0]

	
		for r in range(configurations.model_len - 1, -1, -1):
			# config.split_layer = r
			if r < configurations.model_len - 1:
				self.reinitialize([r], split, first, configurations.LR)

			# print("split point: "+str(r))
			forward_device, forward_server, backward_server, backward_device = self.train(trainloader)
			device_forward_splitwise_latency[r] = forward_device
			server_forward_splitwise_latency[r] = forward_server
			server_backward_splitwise_latency[r] = backward_server
			device_backward_splitwise_latency[r] = backward_device

			if r > 49:
				LR = configurations.LR * 0.1
			
			
		
		print("device_forward "+str(device_forward_splitwise_latency)+"\n server_forward "+str(server_forward_splitwise_latency)+"\ndevice_backward "+ str(device_backward_splitwise_latency)+ "\nserver_backward "+str(server_backward_splitwise_latency))
		#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	
		device_forward_splitwise_latency_temp = [0.009353160858154297, 0.01496434211730957, 0.019944190979003906, 0.02015913009643555, 0.043131608963012695, 0.107067930221557617, 0.12375044822693]
		device_backward_splitwise_latency_temp = [0.01922104835510254, 0.011676549911499023, 0.016455078125, 0.028456230163574219, 0.022606611251831055, 0.090107507705688477, 0.0812466430664062]

		#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		# # # pi 
		# device_forward_splitwise_latency_temp = [0.8614792823791504, 0.9474797248840332, 1.447148323059082, 1.4670040607452393, 1.647247552871704, 1.7771625518798828, 1.8515172004699707]
		# device_backward_splitwise_latency_temp = [0.4919867515563965, 0.5453286170959473, 1.5309937000274658, 1.4755029678344727, 1.9282042980194092, 2.0049049854278564, 2.091542959213257]
		#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		
		transmision_layerwise_latency, receiving_layerwise_latency = self.transmission_layerwise_time(throughput)
		
		device_training_computation_time_array = []
		server_training_computation_time_array = []
		total_training_time_array = []

		device_training_computation_time_array = [(device_forward_splitwise_latency_temp[i] + device_backward_splitwise_latency_temp[i]) for i in range(len(device_forward_splitwise_latency))]
		server_training_computation_time_array = [(server_forward_splitwise_latency[i] + server_backward_splitwise_latency[i]) for i in range(len(device_forward_splitwise_latency))]
		total_training_time_array = [(device_training_computation_time_array[p] + server_training_computation_time_array[p] + transmision_layerwise_latency[p] + receiving_layerwise_latency[p]) for p in range(len(device_forward_splitwise_latency))]

		computation_power, transmission_power, receiving_power = self.measure_power()

		splitwise_computation_energy = [element * computation_power for element in device_training_computation_time_array]
		layerwise_transmission_energy = [element * transmission_power for element in transmision_layerwise_latency] 
		layerwise_receiving_energy = [element * receiving_power for element in receiving_layerwise_latency] 


		total_energy_per_iter_array = []

		total_energy_per_iter_array = [(splitwise_computation_energy[i] + layerwise_transmission_energy[i] + layerwise_receiving_energy[i]) for i in range(len(splitwise_computation_energy))]

		return total_training_time_array, total_energy_per_iter_array

	def ARES_optimiser(self, alpha, throughput):
		print("alpha: " + str(alpha))
		total_training_time_array, total_energy_per_iter_array = self.training_time_energy(throughput)
		
		print("==============================")
		print("total_training_time_array "+str(total_training_time_array)+"\n total_energy_per_iter_array "+str(total_energy_per_iter_array))
		#normalisation
		norm = np.linalg.norm(total_training_time_array)
		normal_training_time_array = total_training_time_array/norm
		
		norm2 = np.linalg.norm(total_energy_per_iter_array)
		normal_energy_per_iter_array = total_energy_per_iter_array/norm2
		
		print("==============================")
		print(""+str(normal_training_time_array))
		print(""+str(normal_energy_per_iter_array))
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
