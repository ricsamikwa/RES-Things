
while True:
    
        
    # current input
    with open('/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/in_current0_input') as t:
        current = ((t.read()))
        print(current)

    # voltage 
    with open('/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/in_voltage0_input') as t:
        current = ((t.read()))
        print(current)

    # power input
    with open('/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/in_power0_input') as t:
        current = ((t.read()))
        print(current)