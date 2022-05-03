import time
import torch
import pickle
import csv
import argparse

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import sys
sys.path.append('../')
from ARES_training.Edge_Server import Edge_Server
import configurations
import functions

parser=argparse.ArgumentParser()
parser.add_argument('--split', help='ARES SPLIT', type= functions.str2bool, default= False)
args=parser.parse_args()

LR = configurations.LR
split = args.split
first = True 
ip_address = '192.168.1.38'

logger.info('Preparing Edge Server.')
Edge_Server = Edge_Server(0, ip_address, configurations.SERVER_PORT, 'VGG')
Edge_Server.initialize(configurations.split_layer, split, first, LR)
first = False

#if split:
	#handle changes of split layers

if split:
	logger.info('ARES Training')
else:
	logger.info('Local Training')

# res = {}
# res['trianing_time'], res['test_acc_record'], res['bandwidth_record'] = [], [], []

for r in range(configurations.R):
	logger.info('====================================>')
	logger.info('==> Round {:} Start'.format(r))

	s_time = time.time()
	bandwidth = Edge_Server.train(thread_number= configurations.K, client_ips= configurations.CLIENTS_LIST)
	new_model = Edge_Server.aggregate(configurations.CLIENTS_LIST)
	e_time = time.time()

	# Recording each round training time, bandwidth and test accuracy
	trianing_time = e_time - s_time
	# res['trianing_time'].append(trianing_time)
	# res['bandwidth_record'].append(bandwidth)

	test_acc = Edge_Server.test(r)
	# res['test_acc_record'].append(test_acc)

	#temp item - WALK - senstive
	# config.split_layer[0] = config.split_layer[0] - 1
	
    #++++++++++++++++++++++++++++++++++++++

	if split:
		# ADAPT SPLIT LAYERS HERE!
		# split_layers = [2]
		# config.split_layer = split_layers
		split_layers = Edge_Server.adaptive_split(bandwidth)
		splitlist = ''.join(str(e) for e in split_layers)
		filename = 'ARES_split_'+splitlist+'_config_3_temp.csv'
	else:
		split_layers = configurations.split_layer
		filename = 'classic_local_config_3_temp.csv'


	with open(configurations.home +'/slogs/'+filename,'a', newline='') as file:
		writer = csv.writer(file)
		writer.writerow([ trianing_time, test_acc])
    
	logger.info('Round Finish')
	logger.info('==> Round Training Time: {:}'.format(trianing_time))

	logger.info('==> Reinitialization for Round : {:}'.format(r + 1))
	
	
	if configurations.split_layer[0] == -1:
		break
	if r > 49:
		LR = configurations.LR * 0.1

	Edge_Server.reinitialize(split_layers, split, first, LR)
	logger.info('==> Reinitialization Finish')

