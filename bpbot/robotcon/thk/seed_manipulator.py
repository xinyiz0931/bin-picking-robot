#!/usr/bin/env python

import sys
import os
import serial
import time

class MySerial(serial.Serial):
    import io

    def __init__(self):
        serial.Serial.__init__(self)
        
    def readline(self):
        ret = ''
        while True:
            c = self.read(1)
            if c == '':
                return ret
            elif c == '\r':
                return ret + c
            else:
                ret += c

## Serial connection class
class SerialConnection(object):
    def __init__(self):
        self.ser = MySerial()
        self.connection = False

    ## Connect serial
    def open_serial(self, port = '/dev/ttyUSB0'):
        self.ser.port         = port
        self.ser.baudrate     = 9600
        self.ser.parity       = serial.PARITY_NONE
        self.ser.bytesize     = serial.EIGHTBITS
        self.ser.stopbits     = serial.STOPBITS_ONE
        self.ser.xonxoff      = False
        self.ser.rtscts       = False
        self.ser.timeout      = 1

        try:
            self.ser.open()
        except serial.SerialException, e:
            print "Could not open serial port%s: %s\n" % (self.ser.portstr, e)
            return False
        else:
            self.ser.flushInput()
            self.ser.flushOutput()
            self.connection = True
            return True

    def close_serial(self):
        if self.connection:
            self.ser.close()

    def write_serial(self, msg):
        if self.connection:
            self.ser.write(msg)

    def flush_input(self):
        if self.connection:
            self.ser.flushInput()

    def readline(self):
        if not self.connection:
            return ""
        timeout = time.time() + self.ser.timeout
        while timeout > time.time():
            if self.ser.inWaiting() > 0:
                return self.ser.readline()
            else:
                time.sleep(0.05)

class SeedConnection(SerialConnection):
    CMD_PREFIX  = 't30'
    CMD_MOVE    = '005F0'
    CMD_INITPOS_L = '03000000'
    CMD_CLOSE_L   = '02000000'
    CMD_OPEN_L    = '01000000'
    CMD_INITPOS_R = '04000000'
    CMD_CLOSE_R   = '03000000'
    CMD_OPEN_R    = '04000000'
    CMD_CTRL1   = '00530'
    CMD_CTRL2   = '00540'
    CMD_HALT    = '01000000'
    CMD_UNLOCK  = '00000000'
    CMD_RESET   = '00000000'
    DATA_LENGTH = '8'
    SENDER_ID   = 'F'
    CR          = '\r'
    
    def __init__(self):
        SerialConnection.__init__(self)

    def generate_command(self, cmd_str1, cmd_str2, id):
        print self.CMD_PREFIX + str(id) + self.DATA_LENGTH + self.SENDER_ID + str(id) + cmd_str1 + str(id) + cmd_str2 + self.CR
        return self.CMD_PREFIX + str(id) + self.DATA_LENGTH + self.SENDER_ID + str(id) + cmd_str1 + str(id) + cmd_str2 + self.CR

    def connect_CAN(self):
        port = '/dev/ttyUSB0'
        if os.path.isfile('.port'):
            f = open('.port', 'r')
            port = f.readline().strip()
            f.close()
        if not self.open_serial(port):
            return False
        self.write_serial("S8\r")
        self.write_serial("O\r")
        self.write_serial("\r")
        self.write_serial("\r")
        self.write_serial("\r")
        return True

    def close_CAN(self):
        self.write_serial("C\r")
        self.close_serial()

    def go_init(self, id):
        self.flush_input()
        if id ==1:
            self.write_serial(self.generate_command(self.CMD_MOVE, self.CMD_INITPOS_L, id))
        else:
            self.write_serial(self.generate_command(self.CMD_MOVE, self.CMD_INITPOS_R, id))

    def close_hand(self, id):
        if id==1:
            self.write_serial(self.generate_command(self.CMD_MOVE, self.CMD_CLOSE_L, id))
        else:
            self.write_serial(self.generate_command(self.CMD_MOVE, self.CMD_CLOSE_R, id))

    def open_hand(self, id):
        if id==1:
            self.write_serial(self.generate_command(self.CMD_MOVE, self.CMD_OPEN_L, id))
        else:
            self.write_serial(self.generate_command(self.CMD_MOVE, self.CMD_OPEN_R, id))

    def halt_hand(self, id):
        self.write_serial(self.generate_command(self.CMD_CTRL1, self.CMD_HALT, id))

    def unlock_hand(self, id):
        self.write_serial(self.generate_command(self.CMD_CTRL1, self.CMD_UNLOCK, id))

    def reset_error(self, id):
        self.write_serial(self.generate_command(self.CMD_CTRL2, self.CMD_RESET, id))

    def getCANUSBversion(self):
        self.flush_input()
        self.write_serial("V\r")
        print self.readline()

    def getCANUSBnumber(self):
        self.flush_input()
        self.write_serial("N\r")
        print self.readline()


class HandState(object):
    STATE_NONE = 0
    STATE_CLOSE_R = 1
    STATE_OPEN_R = 2
    STATE_CLOSE_L = 3
    STATE_OPEN_L = 4

    def __init__(self):
        self.state0 = self.STATE_NONE
        self.state1 = self.STATE_NONE

    def set_closestate(self, id):
        if id == 1:
            self.state0 = self.STATE_CLOSE_L
        else:
            self.state1 = self.STATE_CLOSE_R

    def set_openstate(self, id):
        if id == 1:
            self.state0 = self.STATE_OPEN_L
        else:
            self.state1 = self.STATE_OPEN_R

    def is_closestate(self, id):
        if id == 1:
            return (self.state0 == self.STATE_CLOSE_L)
        else:
            return (self.state1 == self.STATE_CLOSE_R)

    def is_openstate(self, id):
        if id == 1:
            return (self.state0 == self.STATE_OPEN_L)
        else:
            return (self.state1 == self.STATE_OPEN_R)
           
sed_con = None
sed_state = None

def init():
    global sed_con
    global sed_state
    sed_con = SeedConnection()
    sed_con.connect_CAN()
    sed_con.unlock_hand(2)
    sed_con.go_init(2)
    sed_con.unlock_hand(1)
    sed_con.go_init(1)
    sed_state = HandState()

def open_hand(id):
    if sed_con is None:
        return
    if not sed_state.is_openstate(id):
        sed_con.open_hand(id)
        sed_state.set_openstate(id)
    else:
        print "error:hand is already opened"

def close_hand(id):
    if sed_con is None:
        return
    if not sed_state.is_closestate(id):
        sed_con.close_hand(id)
        sed_state.set_closestate(id)
    else:
        print "error:hand is already closed"

if __name__ == '__main__':
   sed = SeedConnection()
   while True:
        command_str = sys.stdin.readline()
        command = command_str.strip()
        if command == 's':
            print 'connect'
            sed.connect_CAN()
        elif command == 'v':
            sed.getCANUSBversion()
        elif command =='u':
            print 'unlock'
            sed.unlock_hand()
        elif command =='h':
            print 'halt'
            sed.halt_hand()
        elif command =='e':
            sed.reset_error()
        elif command == 'i':
            print 'go init'
            sed.go_init()
        elif command == 'o':
            print 'open'
            sed.open_hand()
        elif command == 'c':
            print 'close'
            sed.close_hand()
        elif command == 'q':
            print 'exit'
            sed.close_CAN()
            break
