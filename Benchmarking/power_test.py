
import time
import numpy as np
import sys
import csv

#===================JETSON NANO
power = 0
filename ='comp_MAXN.csv'
while True:
    

    with open('/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/in_power0_input') as t:
        power = ((t.read()))

    with open('./power_logs/' + filename,'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([int(power)])
        

# while True:
    
#     # power input
#     with open('/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/in_power0_input') as t:
#         power = ((t.read()))
#         print(power)


#     # # current input
#     # with open('/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/in_current0_input') as t:
#     #     current = ((t.read()))
#     #     print(current)

#     # # voltage 
#     # with open('/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/in_voltage0_input') as t:
#     #     current = ((t.read()))
#     #     print(current)