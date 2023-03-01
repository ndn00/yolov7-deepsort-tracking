import serial
import time
import json

# arduino = serial.Serial('/dev/ttyACM0', 115200, timeout=5)

# script that simply intermittently writes a message to the Arduino

arduino = serial.Serial(
    port = '/dev/ttyACM0',
    # port='COM3', # change this to macOS port when in the office
    baudrate=115200,
    bytesize=serial.EIGHTBITS,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    timeout=5,
    xonxoff=False,
    rtscts=False,
    dsrdtr=False,
    writeTimeout=2
)

# time.sleep(2)

# def send_count(count):
#     while True:
#         data = arduino.readline().decode("utf-8")
#         # data = arduino.readline()
#         # print(data.decode("utf-8"))
#         print(data + " " + str(type(data)))
#         # if data == "1":
#         #     print(data)
#         #     print("It can read")
#         if data == "1": # data variable sent from Arduino side could be changed to a downlink message eventually
#             print('it can read')
#             arduino.write(json.dumps(count).encode())
#         elif data == "0":
#             print("code zero received")

def send_count(count):
    while True:
        arduino.write(json.dumps(count).encode())
        data = arduino.readline().decode("utf-8")
        print(data)
