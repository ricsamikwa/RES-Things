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
from torchsummary import summary
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity




import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

np.random.seed(0)
torch.manual_seed(0)

class Client(Communicator):
	def __init__(self, index, ip_address, datalen, model_name, split_layer):
		super(Client, self).__init__(index, ip_address)
		self.datalen = datalen
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.model_name = model_name
		self.uninet = utils.get_model('Unit', self.model_name, config.model_len-1, self.device, config.model_cfg)

		# logger.info('Connecting to Server.')
		# self.sock.connect((server_addr,server_port))

	def initialize(self, split_layer, offload, first, LR):
		if offload or first:
			self.split_layer = split_layer

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
				
				# client forward
				forward_time = time.time()	
				self.device_optimizer.zero_grad()
				outputs = self.net(inputs)
				client_output = outputs.clone().detach().requires_grad_(True)
				forward_end_time = time.time()
				
				# print(sys.getsizeof(inputs))
				# print(sys.getsizeof(outputs))
				# print(sys.getsizeof(client_output))

				# server forward/backward
				self.server_optimizer.zero_grad()
				outputs_server = self.server_net(client_output)
				loss = self.criterion(outputs_server, targets)
				server_forward_end_time = time.time()
				
				loss.backward()
				self.server_optimizer.step()
				server_backward_end_time = time.time()

				#client backward
				client_grad = client_output.grad.clone().detach()
				outputs.backward(client_grad)
				self.device_optimizer.step()
				device_backward_end_time = time.time()

				break

		forward_device = forward_end_time - forward_time
		forward_server = server_forward_end_time - forward_end_time
		backward_server = server_backward_end_time - server_forward_end_time
		backward_device = device_backward_end_time - server_backward_end_time

		if self.split_layer == 6:
			forward_device = forward_end_time - forward_time
			backward_device = device_backward_end_time - forward_end_time
			forward_server = 0
			backward_server = 0


		return forward_device, forward_server, backward_server, backward_device
		
	def upload(self):
		msg = ['MSG_LOCAL_WEIGHTS_CLIENT_TO_SERVER', self.net.cpu().state_dict()]
		self.send_msg(self.sock, msg)

	def reinitialize(self, split_layers, offload, first, LR):
		self.initialize(split_layers, offload, first, LR)

logger.info('Preparing Client')
client = Client(1, '192.168.5.22', 50000, 'VGG5', 6)

offload = False
first = True # First initializaiton control
# first = True # First initializaiton control

client.initialize(6, offload, first, config.LR)
# first = False 
first = True 

logger.info('Preparing Data.')
# this has problems on mac
cpu_count = multiprocessing.cpu_count()
trainloader, classes= utils.get_local_dataloader(1, 0)


device_forward_splitwise_latency = [0,0,0,0,0,0]
server_forward_splitwise_latency = [0,0,0,0,0,0]
server_backward_splitwise_latency = [0,0,0,0,0,0]
device_backward_splitwise_latency = [0,0,0,0,0,0]

config.split_layer = 5
for r in range(config.model_len - 1, 0, -1):
	print(r)
	print(config.split_layer)
	forward_device, forward_server, backward_server, backward_device = client.train(trainloader)
	device_forward_splitwise_latency[r -1] = forward_device
	server_forward_splitwise_latency[r -1] = forward_server
	server_backward_splitwise_latency[r -1] = backward_server
	device_backward_splitwise_latency[r -1] = backward_device

	if r > 49:
		LR = config.LR * 0.1
	config.split_layer = r
	if r < config.model_len - 1:
		client.reinitialize(r, offload, first, config.LR)
	# logger.info('==> Reinitialization Finish')

print(str(device_forward_splitwise_latency)+"\n"+str(server_forward_splitwise_latency)+"\n"+ str(device_backward_splitwise_latency)+ "\n"+str(server_backward_splitwise_latency))