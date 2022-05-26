#some information here


#example input:
power_control <Device> <Desired_State>
power_control central_ecu on
power_control ivi off
power_control hog restart

$ power_control hog on
[power_control] powering on “hog”....
[power_control] command issued
[power_control] “hog” power on completed.
For example, the complete sequence for a powering on a relay that it will run through:
<connect to serial port>
Input: ‘ver\n’
Output: ‘00000008\n>’ #Check this against the configured firmware version
Input: ‘id\n’
Output: ‘12A45D\n>’ #Check this against the configured board id
Input: ‘relay read 0\n’
Output: ‘off\n>’ #If the relay is off, turn it on. If it’s already on then no action needed.
Input: ‘relay on 0\n’
Output : ‘\n>’
Input: ‘Relay read 0\n’ #Verify the relay turned on as expected
Output: ‘on\n>’

#Relay control for Numato Relay 
#Version 1.0
#Written by Savion Ragster
#This program controls the relay - switchign power to the hog
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