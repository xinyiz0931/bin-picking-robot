from nextage_ros_bridge import nextage_client
from hrpsys import rtm
def pred():
    global robot
    host = '192.168.128.10'
    port = '15005'
    robot_name = "RobotHardware0"
    print('host:' + host)
    print('port:' + port)
    
    rtm.nshost = host
    rtm.nsport = port

    robot = nxc = nextage_client.NextageClient()
    robot.init(robotname=robot_name, url="")
    return robot