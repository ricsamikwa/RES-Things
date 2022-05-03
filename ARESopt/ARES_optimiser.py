
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import threading
import tqdm
import time
import random
import numpy as np
import sys
sys.path.append('../')
from ARESopt.ARES_optimisation import BenchClient

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


from Wireless import *



logger.info('Preparing Server - Device')

benchClient = BenchClient(1, '192.168.1.100', 50000, 'VGG', 6)
temp_throughput = 15

spliting_strategy = benchClient.ARES_optimiser(0.5, temp_throughput) +1
print("Current SPLIT strategy: "+ str(spliting_strategy))
