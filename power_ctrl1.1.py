#Relay control for Numato Relay 
#Version 1.1
#Written by Savion Ragster
#This program controls the relay - switchign power to devices.
#This program will be on jump host

#communication usb serial connection
import time



#devices:
#hog, ivi, central_ecu
#states:
#on, off, restart 
def power_ctrlf(device, state, delay = 5):

    #device mapping:
    if device == 'hog':
        relay_no = 
    elif device == 'ivi':
        relay_no = 
    elif device == 'central_ecu'
        relay_no = 

  #connect to serial port:


   if state == 'on':  #turn on hardware:
        print(f'device {device} on')
        relay on relay_no
        relay read relay_no
  
    if cmd == 'off':  #turn off hardware:
        print(f'device {device} off')
        relay off relay_no
        relay read relay_no

    if cmd == 'restart': #restart hardware:
        print(f'device {device} restarting..')
        relay off relay_no
        relay read relay_no
        time.sleep(delay)
        relay on relay_no
        relay read relay_no
    return device, state


    #questions - should it wait and check info from hog to verify that it's on? or send some updates on state?
    #how do I find the realay numbers?