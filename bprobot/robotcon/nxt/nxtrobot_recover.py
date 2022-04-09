import sys
sys.path.append("./")
from robotcon.nxt.nxtrobot_client import NxtRobot

if __name__ == "__main__":
    nxt = NxtRobot(host='[::]:15005')
    nxt.servoOff()
    nxt.servoOn()
    nxt.goInitial()
