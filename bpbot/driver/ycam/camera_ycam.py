import paramiko
import os
import sys

class RemoteCamera:
    def __init__(self):
        self.__host = '100.80.147.231'
        self.__port = 22
        self.__user = 'xinyi'
        self.__pass = 'st'
        self.__connected = False

    def connect_shh(self):
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(hostname=self.__host, 
                        port=self.__port, 
                        username=self.__user, 
                        password=self.__pass)
            print("Romote camera connected! ")
            self.__connect = True
            sftp = ssh.open_sftp()
            return sftp
        except:
            print("Fail to connect camera! ")
    
    def capture(self, c_path):
        # Extremely silly solution. Need to be improved
        print("-------------------------------------")
        os.system(c_path)
        print("-------------------------------------")
    
            
    def receive_file(self, remote_path, local_path, sftp):
        if self.__connect:
            # sftp = ssh.open_sftp()
            print ("Receive remote point cloud! ")
            sftp.get(remote_path, local_path)
            # ssh.close()
        else:
            print("Fail to receive file! ")
            # ssh.close()

if __name__ == "__main__":
    rc = RemoteCamera()
    rc.connect_shh()
