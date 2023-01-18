import numpy as np
fz = 5
tx = 0.02613765
ty = -0.03972532
t_thld = 0.2
r_min = 30
r_max = 80
j3 = np.sign(ty)*((np.abs(ty)/t_thld) * (r_max-r_min) + r_min)
j4 = 0 if tx > 0 else ((tx/t_thld) * (r_max-r_min) + r_min)
# j4 = ((np.abs(tx)/t_thld) * (r_max-r_min) + r_min)
print("j3=",j3, "j4=",j4)
from bpbot.motion import FlingActor
actor = FlingActor()
actor.add_fling_action(j3=j3, j4=j4, h=1)