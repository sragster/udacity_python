#Relay control for Numato Relay 
#Version 1.0
#Written by Savion Ragster
#This program controls the relay - switchign power to the hog
#This program will be on jump host

#communication usb serial connection


#commands to control relay:
relay on 0#turs relay 0 on
relay off 0, 
relay read 0#returns state?

#To send commands - use API equaialent to write/writefile pass the buffer/string continaing the command. 
#carriage return ASCIII 13  to emulate enter key 
#if you need return - read from cerial port buffer. 


#psydocode from JavaScript example:
SerialPort = require("serialport").SerialPort
//On Windows use the port name such as COM4 and on Linux/Mac, use the device node name such as /dev/ttyACM0
var port = "com4";

var portObj = new SerialPort(port,{
 baudrate: 19200
}, false);
portObj.open(function (error){
  if ( error ) {
		console.log('Failed to open port: ' + error);
  } else {
   //Communicate with the device
}
portObj.on('data', function(data){
	console.log('Data Returned by the device');
	console.log('--------------------');
        console.log(String(data));
	console.log('--------------------');
});

var SerialPort = require("serialport").SerialPort
var port = "COM11";

var portObj = new SerialPort(port,{
  baudrate: 19200
}, false);

portObj.on('data', function(data){
	console.log('Data Returned by the device');
	console.log('--------------------');
        console.log(String(data));
	console.log('--------------------');
        portObj.close();
});

portObj.open(function (error){
  if ( error ) {
		console.log('Failed to open port: ' + error);
  } else {
		console.log('Writing command gpio set 0 to port');
		portObj.write("gpio set 0r", function(err, results){
			if(error){
				console.log('Failed to write to port: '+ error);
			}
		});
		
		console.log('Waiting for two seconds');
		setTimeout(
			function(){
				console.log('Writing command gpio clear 0 to port');
				portObj.write("gpio clear 0r", function(err, results){
					if(error){
						console.log('Failed to write to port: '+ error);
					}
				});
				
				setTimeout( function(){process.exit(code=0);}, 1000);
			}
		,2000);
  }
});