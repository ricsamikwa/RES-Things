

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import threading
import tqdm
import time
import random
import numpy as np
from ARESopt.ARES_optimisation import BenchClient

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import sys
sys.path.append('../')
from Wireless import *
import functions
import configurations

np.random.seed(0)
torch.manual_seed(0)

class Edge_Server(Wireless):
	def __init__(self, index, ip_address, server_port, model_name):
		super(Edge_Server, self).__init__(index, ip_address)
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.port = server_port
		self.model_name = model_name
		self.sock.bind((self.ip, self.port))
		self.client_socks = {}

		while len(self.client_socks) < configurations.K:
			self.sock.listen(5)
			logger.info("CONNECTIONS.")
			(client_sock, (ip, port)) = self.sock.accept()
			logger.info('connection ' + str(ip))
			logger.info(client_sock)
			self.client_socks[str(ip)] = client_sock

		self.uninet = functions.get_model('Unit', self.model_name, configurations.model_len-1, self.device, configurations.model_cfg)

		self.transform_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
		self.testset = torchvision.datasets.CIFAR10(root=configurations.dataset_path, train=False, download=False, transform=self.transform_test)
		self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=100, shuffle=False, num_workers=4)
 		
	def initialize(self, split_layers, offload, first, LR):
		if offload or first:
			self.split_layers = split_layers
			self.nets = {}
			self.optimizers= {}
			for i in range(len(split_layers)):
				client_ip = configurations.CLIENTS_LIST[i]
				if split_layers[i] < len(configurations.model_cfg[self.model_name]) -1: # Only offloading client need initialize optimizer in server
					self.nets[client_ip] = functions.get_model('Server', self.model_name, split_layers[i], self.device, configurations.model_cfg)

					#offloading weight in server also need to be initialized from the same global weight
					cweights = functions.get_model('Client', self.model_name, split_layers[i], self.device, configurations.model_cfg).state_dict()
					pweights = functions.split_weights_server(self.uninet.state_dict(),cweights,self.nets[client_ip].state_dict())
					self.nets[client_ip].load_state_dict(pweights)

					self.optimizers[client_ip] = optim.SGD(self.nets[client_ip].parameters(), lr=LR,
					  momentum=0.9)
				else:
					self.nets[client_ip] = functions.get_model('Server', self.model_name, split_layers[i], self.device, configurations.model_cfg)
			self.criterion = nn.CrossEntropyLoss()

		msg = ['MSG_INITIAL_GLOBAL_WEIGHTS_SERVER_TO_CLIENT', self.uninet.state_dict()]
		for i in self.client_socks:
			self.send_msg(self.client_socks[i], msg)

	def train(self, thread_number, client_ips):
		# Network test
		self.net_threads = {}
		for i in range(len(client_ips)):
			self.net_threads[client_ips[i]] = threading.Thread(target=self._thread_network_testing, args=(client_ips[i],))
			self.net_threads[client_ips[i]].start()

		for i in range(len(client_ips)):
			self.net_threads[client_ips[i]].join()

		self.bandwidth = {}
		for s in self.client_socks:
			msg = self.recv_msg(self.client_socks[s], 'MSG_TEST_NETWORK')
			self.bandwidth[msg[1]] = msg[2]

		# Training start
		self.threads = {}
		for i in range(len(client_ips)):
			if configurations.split_layer[i] == (configurations.model_len -1):
				self.threads[client_ips[i]] = threading.Thread(target=self._thread_training_no_offloading, args=(client_ips[i],))
				logger.info(str(client_ips[i]) + 'training start')
				self.threads[client_ips[i]].start()
			else:
				logger.info(str(client_ips[i]))
				self.threads[client_ips[i]] = threading.Thread(target=self._thread_training_offloading, args=(client_ips[i],))
				logger.info(str(client_ips[i]) + 'training start')
				self.threads[client_ips[i]].start()

		for i in range(len(client_ips)):
			self.threads[client_ips[i]].join()

		self.ttpi = {} # Training time per iteration
		for s in self.client_socks:
			msg = self.recv_msg(self.client_socks[s], 'MSG_TIME_ITERATION')
			self.ttpi[msg[1]] = msg[2]

		return self.bandwidth

	def _thread_network_testing(self, client_ip):
		msg = self.recv_msg(self.client_socks[client_ip], 'MSG_TEST_NETWORK')
		msg = ['MSG_TEST_NETWORK', self.uninet.cpu().state_dict()]
		self.send_msg(self.client_socks[client_ip], msg)

	def _thread_training_no_offloading(self, client_ip):
		pass

	def _thread_training_offloading(self, client_ip):
		#issues here!!
		# iteration = int((config.N / (config.K * config.B)))
		iteration = 50 # verify this number 50000/(5*100) = 100, but we have 50 iterations from the data ?
		# logger.info(str(iteration) + ' iterations!!')
		for i in range(iteration):
			msg = self.recv_msg(self.client_socks[client_ip], 'MSG_INTERMEDIATE_ACTIVATIONS_CLIENT_TO_SERVER')
			smashed_layers = msg[1]
			labels = msg[2]
			# logger.info(' received smashed data !!')
			inputs, targets = smashed_layers.to(self.device), labels.to(self.device)
			self.optimizers[client_ip].zero_grad()
			outputs = self.nets[client_ip](inputs)
			loss = self.criterion(outputs, targets)
			loss.backward()
			self.optimizers[client_ip].step()

			msg = ['MSG_INTERMEDIATE_GRADIENTS_SERVER_TO_CLIENT_'+str(client_ip), inputs.grad]
			self.send_msg(self.client_socks[client_ip], msg)

		logger.info(str(client_ip) + 'training end')
		return 'Done'

	def aggregate(self, client_ips):
		w_local_list =[]
		for i in range(len(client_ips)):
			msg = self.recv_msg(self.client_socks[client_ips[i]], 'MSG_SUB_WEIGHTS_CLIENT_TO_SERVER')
			if configurations.split_layer[i] != (configurations.model_len -1):
				w_local = (functions.concat_weights(self.uninet.state_dict(),msg[1],self.nets[client_ips[i]].state_dict()),configurations.N / configurations.K)
				w_local_list.append(w_local)
			else:
				w_local = (msg[1],configurations.N / configurations.K)
				w_local_list.append(w_local)
		zero_model = functions.zero_init(self.uninet).state_dict()
		aggregrated_model = functions.fed_avg(zero_model, w_local_list, configurations.N)
		
		self.uninet.load_state_dict(aggregrated_model)
		return aggregrated_model

	def test(self, r):
		self.uninet.eval()
		test_loss = 0
		correct = 0
		total = 0
		with torch.no_grad():
			for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(self.testloader)):
				inputs, targets = inputs.to(self.device), targets.to(self.device)
				outputs = self.uninet(inputs)
				loss = self.criterion(outputs, targets)

				test_loss += loss.item()
				_, predicted = outputs.max(1)
				total += targets.size(0)
				correct += predicted.eq(targets).sum().item()

		acc = 100.*correct/total
		logger.info('Test Accuracy: {}'.format(acc))

		# Save checkpoint.
		torch.save(self.uninet.state_dict(), './'+ configurations.model_name +'.pth')

		return acc

	# The function to change more
	def adaptive_split(self, bandwidth):
		
		logger.info('Preparing Device')
		benchClient = BenchClient(1, '192.168.1.100', 50000, 'VGG', 6)

		offloading_strategy = benchClient.ARES_optimiser(0.6, bandwidth[configurations.CLIENTS_LIST[0]]) + 1
		print("Current Strategy: "+ str(offloading_strategy))
		# strategy configuration - refactoring
		configurations.split_layer = [1,3,4,5,5]
		logger.info('Next Round : ' + str(configurations.split_layer))

		msg = ['SPLIT_VECTOR',configurations.split_layer]
		self.scatter(msg)
		return configurations.split_layer


	def reinitialize(self, split_layers, offload, first, LR):
		self.initialize(split_layers, offload, first, LR)

	def scatter(self, msg):
		for i in self.client_socks:
			self.send_msg(self.client_socks[i], msg)
