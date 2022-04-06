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
from Sever import Sever
import config
import utils
#import PPO

parser=argparse.ArgumentParser()
parser.add_argument('--offload', help='ARES or classic local mode', type= utils.str2bool, default= False)
args=parser.parse_args()

LR = config.LR
offload = args.offload
first = True # First initializaiton control
ip_address = '192.168.1.38'
# ip_address = '192.168.0.175'

logger.info('Preparing Sever.')
sever = Sever(0, ip_address, config.SERVER_PORT, 'VGG5')
sever.initialize(config.split_layer, offload, first, LR)
first = False

state_dim = 2*config.G
action_dim = config.G

#if offload:
	#handle changes of split layers

if offload:
	logger.info('ARES Training')
else:
	logger.info('Classic Local (FL) Training')

# res = {}
# res['trianing_time'], res['test_acc_record'], res['bandwidth_record'] = [], [], []

for r in range(config.R):
	logger.info('====================================>')
	logger.info('==> Round {:} Start'.format(r))

	s_time = time.time()
	state, bandwidth = sever.train(thread_number= config.K, client_ips= config.CLIENTS_LIST)
	aggregrated_model = sever.aggregate(config.CLIENTS_LIST)
	e_time = time.time()

	# Recording each round training time, bandwidth and test accuracy
	trianing_time = e_time - s_time
	# res['trianing_time'].append(trianing_time)
	# res['bandwidth_record'].append(bandwidth)

	test_acc = sever.test(r)
	# res['test_acc_record'].append(test_acc)

	#temp item - WALK
	config.split_layer[0] = config.split_layer[0] - 1
	
    #++++++++++++++++++++++++++++++++++++++

	if offload:
		# ADAPT SPLIT LAYERS HERE!
		# split_layers = [2]
		# config.split_layer = split_layers
		split_layers = sever.adaptive_offload(bandwidth)
		splitlist = ''.join(str(e) for e in split_layers)
		filename = 'ARES_split_'+splitlist+'_config_3_temp.csv'
	else:
		split_layers = config.split_layer
		filename = 'classic_local_config_3_temp.csv'


	with open(config.home +'/results/'+filename,'a', newline='') as file:
		writer = csv.writer(file)
		writer.writerow([ trianing_time, test_acc])
    
	logger.info('Round Finish')
	logger.info('==> Round Training Time: {:}'.format(trianing_time))

	logger.info('==> Reinitialization for Round : {:}'.format(r + 1))
	
	
	if config.split_layer[0] == -1:
		break
	if r > 49:
		LR = config.LR * 0.1

	sever.reinitialize(split_layers, offload, first, LR)
	logger.info('==> Reinitialization Finish')

