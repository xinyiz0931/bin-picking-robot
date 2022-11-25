def NestedDictValues(d):
  for v in d.values():
    if isinstance(v, dict):
      yield from NestedDictValues(v)
    else:
        if isinstance(v, np.float64):
            yield float(v)
        else:
            yield v

from bpbot.config import BinConfig
bincfg = BinConfig()
cfg = bincfg.data
import numpy as np

a = np.array([2.1], dtype=np.float64)
print(isinstance(a[0], np.floating))
cfg["pick"]["height"]["min"] = a[0]
for i in NestedDictValues(cfg):
    print(i, isinstance(i, np.floating))
bincfg.write()
# for i in NestedDictValues(cfg):
#     if isinstance(i, np.floating): 
#         print("Still here! ")

        