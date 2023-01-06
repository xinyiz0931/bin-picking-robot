#!/bin/bash
if pgrep -x "pyscript.py" > /dev/null
then
	killall -9 "pyscript.py"
	#kill -9 `ps aux | grep pyscript.py | awk '{print $2}'`
    echo "fount it!"
else
    echo "doesn't exist! "
fi 

