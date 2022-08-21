# bpbot

Bin picking tools including vision, grasp planning and robot control

```
$ git clone https://github.com/xinyiz0931/bin-picking-robot.git
$ cd bin-picking-robot
$ pip install -e .
```

Use bin picking functions by
```
>>> import bpbot
```

e.g. executing files in `examples`
```
$ python example/example_test_graspability.py
```
## Entanglement Map

1. Run this python script to visualize the results
```
$ git clone https://github.com/xinyiz0931/bin-picking-robot.git
$ cd bin-picking-robot
$ pip install -e .
$ python example/example_tangle_pick.py
```
2. Before execution, there are four important parameters must be tuned, you can revise them in `./cfg/config.yaml`. Default values are pretty good, you can tune them based on your needs. 

- (len_thld, dist_thld, sliding_size, sliding_stride, c_size) = t_params
  - len_thld: the minimum length of detected edge segments
  - dist_thld: the minimum distance between detected edge segments
  - sliding_size: (important) size of sliding window
  - sliding_stride: (important) stride between two windows
  - c_size: size of input cropped image when computing entanglement map

3. If you want to revise the source code, please find them in 

- `./bpbot/binpicking.py -> detect_nontangle_grasp()`
- `./bpbot/tangle_solution/entanglement_map.py, topo_coor.py`

