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
			logger.debug(self.net)
			self.criterion = nn.CrossEntropyLoss()

		self.optimizer = optim.SGD(self.net.parameters(), lr=LR,
					  momentum=0.9)
		logger.debug('Receiving Weights..')

	def train(self, trainloader):
	
		s_time_total = time.time()
		time_training_c = 0
		self.net.to(self.device)
		self.net.train()
		if self.split_layer == (config.model_len -1):
			for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(trainloader)):
				inputs, targets = inputs.to(self.device), targets.to(self.device)
				self.optimizer.zero_grad()
				outputs = self.net(inputs)
				loss = self.criterion(outputs, targets)
				loss.backward()
				self.optimizer.step()
			
		else: 
			for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(trainloader)):
				inputs, targets = inputs.to(self.device), targets.to(self.device)
				self.optimizer.zero_grad()
				outputs = self.net(inputs)

				msg = ['MSG_LOCAL_ACTIVATIONS_CLIENT_TO_SERVER', outputs.cpu(), targets.cpu()]
				self.send_msg(self.sock, msg)
				gradients = self.recv_msg(self.sock)[1].to(self.device)

				outputs.backward(gradients)
				self.optimizer.step()

		e_time_total = time.time()
		logger.info('Total time: ' + str(e_time_total - s_time_total))

		training_time_pr = (e_time_total - s_time_total) / int((config.N / (config.K * config.B)))
		logger.info('training_time_per_iteration: ' + str(training_time_pr))


		return e_time_total - s_time_total
		
	def upload(self):
		msg = ['MSG_LOCAL_WEIGHTS_CLIENT_TO_SERVER', self.net.cpu().state_dict()]
		self.send_msg(self.sock, msg)

	def reinitialize(self, split_layers, offload, first, LR):
		self.initialize(split_layers, offload, first, LR)

logger.info('Preparing Client')
client = Client(1, '192.168.5.22', 50000, 'VGG5', 6)

offload = False
first = True 
client.initialize(6, offload, first, config.LR)
first = False 

logger.info('Preparing Data.')
# this has problems on mac OS
cpu_count = multiprocessing.cpu_count()
trainloader, classes= utils.get_local_dataloader(1, cpu_count)

for r in range(config.R):
	logger.info('====================================>')
	logger.info('ROUND: {} START'.format(r))
	training_time = client.train(trainloader)
	logger.info('ROUND: {} END'.format(r))
	
	logger.info('==> Waiting for aggregration')

	logger.info('==> Reinitialization for Round : {:}'.format(r + 1))
	s_time_rebuild = time.time()
	if offload:
		config.split_layer = client.recv_msg(client.sock)[1]

	if r > 49:
		LR = config.LR * 0.1

	client.reinitialize(config.split_layer[0], offload, first, config.LR)
	e_time_rebuild = time.time()
	logger.info('Rebuild time: ' + str(e_time_rebuild - s_time_rebuild))
	logger.info('==> Reinitialization Finish')