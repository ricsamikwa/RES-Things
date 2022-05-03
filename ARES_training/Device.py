import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import time
import numpy as np
import sys
import csv

sys.path.append('../')
import configurations
import functions
from Wireless import *
from threading import Thread

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

np.random.seed(0)
torch.manual_seed(0)
hostname = socket.gethostname().replace('-desktop', '')
ip_address = configurations.HOST2IP[hostname]
index = configurations.CLIENTS_CONFIG[ip_address]

class Client(Wireless):
	def __init__(self, index, ip_address, server_addr, server_port, datalen, model_name, split_layer):
		super(Client, self).__init__(index, ip_address)
		self.datalen = datalen
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.model_name = model_name
		self.uninet = functions.get_model('Unit', self.model_name, configurations.model_len-1, self.device, configurations.model_cfg)

		logger.info('Connecting to Edge.')
		self.sock.connect((server_addr,server_port))

	def initialize(self, split_layer, offload, first, LR):
		if offload or first:
			self.split_layer = split_layer

			logger.debug('Building Model.')
			self.net = functions.get_model('Client', self.model_name, self.split_layer, self.device, configurations.model_cfg)
			logger.debug(self.net)
			self.criterion = nn.CrossEntropyLoss()

		self.optimizer = optim.SGD(self.net.parameters(), lr=LR,
					  momentum=0.9)
		logger.debug('Weights..')
		weights = self.recv_msg(self.sock)[1]
		if self.split_layer == (configurations.model_len -1):
			self.net.load_state_dict(weights)
		else:
			pweights = functions.split_weights_client(weights,self.net.state_dict())
			self.net.load_state_dict(pweights)
		logger.debug('Initialize Finished')

	def power_monitor_thread(self, stop):
		power = 0
		# power input
		filename =''+ hostname+'-'+str(configurations.split_layer[index])+'_power_config_3_temp.csv'
		time.sleep(0.4)
		# while True:
		for x in range(10):
			
			if stop():
				break

			with open('/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/in_power0_input') as t:
				power = ((t.read()))

			# print(power)	
			with open(configurations.home + '/slogs/' + filename,'a', newline='') as file:
				writer = csv.writer(file)
				writer.writerow([int(power)])
				
			time.sleep(0.4)

		return
 
	def train(self, trainloader, hostname):
		# Network speed test
		network_time_start = time.time()
		msg = ['MSG_TEST_NETWORK', self.uninet.cpu().state_dict()]
		# print("message size: "+ str(sys.sizeof(msg)))
		self.send_msg(self.sock, msg)
		msg = self.recv_msg(self.sock,'MSG_TEST_NETWORK')[1]
		network_time_end = time.time()

		# send a model to and from a server (size in bytes * 8 * 2)
		network_speed = (2 * configurations.model_size * 8) / (network_time_end - network_time_start) #Mbit/s 

		logger.info('Network speed is {:}'.format(network_speed))
		msg = ['MSG_TEST_NETWORK', self.ip, network_speed]
		self.send_msg(self.sock, msg)


		# print(hostname[0:3])
		# if hostname[0:4] == 'nano':
		# 	# print('this is a nano')
		# 	stop_threads = False
		# 	t1 = Thread(target=self.power_monitor_thread, args =(lambda : stop_threads,))
		# 	t1.start()
   
		# Training start
		s_time_total = time.time()
		time_training_c = 0
		self.net.to(self.device)
		self.net.train()
		time_tota_temp = 0

		iteration_count = 0
		nice_flag = False

		if self.split_layer == (configurations.model_len -1): # Classic local training
			for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(trainloader)):
				inputs, targets = inputs.to(self.device), targets.to(self.device)
				self.optimizer.zero_grad()
				outputs = self.net(inputs)
				loss = self.criterion(outputs, targets)
				loss.backward()
				self.optimizer.step()
				iteration_count+=1
			
		else: # Split learning
			# print(enumerate(tqdm.tqdm(trainloader)).size)
			for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(trainloader)):
				inputs, targets = inputs.to(self.device), targets.to(self.device)
				self.optimizer.zero_grad()
				outputs = self.net(inputs)

				msg = ['MSG_INTERMEDIATE_ACTIVATIONS_CLIENT_TO_SERVER', outputs.cpu(), targets.cpu()]
				
				self.send_msg(self.sock, msg)
				# print(e_time_tota_temp - s_time_tota_temp)

				# logger.info('waiting to receive gradients')

				# Wait receiving server gradients
				# s_time_tota_temp = time.time()
				gradients = self.recv_msg(self.sock)[1].to(self.device)
				# time_tota_temp += time.time() - s_time_tota_temp
				
				# temp stuff
				if nice_flag:
					print(outputs.shape)
					print(gradients.shape)
					nice_flag = False

				outputs.backward(gradients)
				self.optimizer.step()
				iteration_count+=1
    
		e_time_total = time.time()
		logger.info('Total time: ' + str(e_time_total - s_time_total))

		iteration = int((configurations.N / (configurations.K * configurations.B)))
		# this is a critical
		iteration = 50 # verify this number 50000/(5*100) = 100, but we have 50 iterations from the data ?
		logger.info(str(iteration_count) + ' iterations!!')


		training_time_pr = (e_time_total - s_time_total) /iteration
		average_time = time_tota_temp /iteration
		logger.info('training_time_per_iteration: ' + str(training_time_pr))
		# logger.info('average_receiving_time: ' + str(average_time))

		msg = ['MSG_TIME_ITERATION', self.ip, training_time_pr]
		self.send_msg(self.sock, msg)
  
		# if hostname[0:3] == 'nano':
		# 	stop_threads = True
		# 	t1.join()
		# 	print('thread killed')

		return e_time_total - s_time_total, training_time_pr, network_speed, average_time
		
	def upload(self):
		msg = ['MSG_SUB_WEIGHTS_CLIENT_TO_SERVER', self.net.cpu().state_dict()]
		self.send_msg(self.sock, msg)

	def reinitialize(self, split_layers, offload, first, LR):
		self.initialize(split_layers, offload, first, LR)

