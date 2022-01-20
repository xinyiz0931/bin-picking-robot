import os

print(os.path.abspath(__file__))
curdir = os.path.dirname(os.path.abspath(__file__))
print(curdir)
