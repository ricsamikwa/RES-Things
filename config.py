import sys

SERVER_ADDR= '192.168.1.38'
# SERVER_ADDR= '192.168.0.175'
SERVER_PORT = 51000


# Everyone
# HOST2IP = {'pi':'192.168.1.33' , 'nano2':'192.168.1.41', 'nano4':'192.168.1.40' , 'nano6':'192.168.1.42', 'nano8':'192.168.1.43'}
# CLIENTS_CONFIG= {'192.168.1.33':0, '192.168.1.41':1, '192.168.1.40':2, '192.168.1.42':3, '192.168.1.43':4}
# CLIENTS_LIST= ['192.168.1.33', '192.168.1.41', '192.168.1.40', '192.168.1.42', '192.168.1.43'] 
G = 1
# #Three Clients
K = 2 # Number of devices

HOST2IP = {'nano6':'192.168.1.42', 'nano8':'192.168.1.43'}
CLIENTS_CONFIG= { '192.168.1.42':0, '192.168.1.43':1}
CLIENTS_LIST= [ '192.168.1.42', '192.168.1.43'] 

# The initial one
# K = 1 # Number of devices

# HOST2IP = {'nano6':'192.168.1.42' }
# CLIENTS_CONFIG= {'192.168.1.42':0 }
# CLIENTS_LIST= ['192.168.1.42'] 

# HOST2IP = {'nano8':'192.168.1.43' }
# CLIENTS_CONFIG= {'192.168.1.43':0 }
# CLIENTS_LIST= ['192.168.1.43'] 
# # Pi only

# HOST2IP = {'raspberrypi':'192.168.1.44' }
# CLIENTS_CONFIG= {'192.168.1.44':0 }
# CLIENTS_LIST= ['192.168.1.44'] 

# HOST2IP = {'raspberrypi':'192.168.0.105' }
# CLIENTS_CONFIG= {'192.168.0.105':0 }
# CLIENTS_LIST= ['192.168.0.105'] 

# Dataset configration
dataset_name = 'CIFAR10'
home = sys.path[0].split('RES-Things')[0] + 'RES-Things'
dataset_path = home +'/datasets/'+ dataset_name +'/'
N = 50000 # data length


# Model configration
model_cfg = {
	# (Type, in_channels, out_channels, kernel_size, out_size(c_out*h*w), flops(c_out*h*w*k*k*c_in))
	'VGG5' : [('C', 3, 32, 3, 32*32*32, 32*32*32*3*3*3), ('M', 32, 32, 2, 32*16*16, 0), 
	('C', 32, 64, 3, 64*16*16, 64*16*16*3*3*32), ('M', 64, 64, 2, 64*8*8, 0), 
	('C', 64, 64, 3, 64*8*8, 64*8*8*3*3*64), 
	('D', 8*8*64, 128, 1, 64, 128*8*8*64), 
	('D', 128, 10, 1, 10, 128*10)]
}
model_name = 'VGG5'
model_size = 1.28
model_flops = 32.902
total_flops = 8488192
# split_layer = [2, 3, 2] #Initial split layers - with offloading
split_layer = [6, 6] #Initial split layers no offloading
# split_layer = [6] #Initial split layer for one
model_len = 7

R = 100 # rounds
LR = 0.01 # learning rate
B = 100 # minibatch size

iteration = {'192.168.1.33':5, '192.168.1.41':5, '192.168.1.40':10, '192.168.1.42':5, '192.168.1.43':5}  # infer times for each device

random = True
random_seed = 0