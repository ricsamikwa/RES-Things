import numpy as np
import time
# import socket
import pickle
import json
import socket
import sys

part_output=np.random.randint(10,90,(255,255))
# print(part_output)
print("Data Size: ",sys.getsizeof(part_output))

for x in range(1000):
 
  client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

  # SERVER - RECEIVING
  client.connect(('192.168.1.42', 8080))

  # DEVICE - RECEIVING
  # client.connect(('192.168.1.43', 8080))

  data_part=pickle.dumps(part_output, protocol=pickle.HIGHEST_PROTOCOL)
  client.sendall(data_part)