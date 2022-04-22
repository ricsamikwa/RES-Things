
import numpy as np
import time
# import socket
import pickle
import socket


partition_point = 1
BUFFER_SIZE = 4096

print("Waiting for data (IoT device) =====> (Server)")
# On the server
time.sleep(10)
count = 1
serv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# SERVER - RECEIVING
serv.bind(('192.168.1.38', 8080))
# DEVICE - RECEIVING
# serv.bind(('192.168.1.43', 8080))

serv.listen(5)
while True:
    conn, addr = serv.accept()
    data=[]
    
    while True:
        while 1:
            tensor = conn.recv(BUFFER_SIZE)
            if not tensor: break
            # data.append(tensor)      
        break
    conn.close()
    # print('client disconnected')





