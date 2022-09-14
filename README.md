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
# A Topological Solution of Entanglement for Complex-shaped Parts in Robotic Bin-picking

[Xinyi Zhang](http://xinyiz0931.github.io), [Keisuke Koyama](https://kk-hs-sa.website/), Yukiyasu Domae, [Weiwei Wan](https://wanweiwei07.github.io/) and [Kensuke Harada](https://www.roboticmanipulation.org/members2/kensuke-harada/)    
Osaka University     
IEEE International Conference on Automation Science and Engineering (CASE 2021)

[arXiv](https://arxiv.org/pdf/2106.00943.pdf) / [Video](https://www.youtube.com/watch?v=5WTpQAjoArM)

## Overview

<img src="https://xinyiz0931.github.io/images/project_emap_teaser.jpg" width="60%" >

This paper addresses the problm of picking up only one object at a time avoiding any entanglement in bin-picking. To cope with a difficult case where the complex-shaped objects are heavily entangled together, we propose a topology-based method that can generate non-tangle grasp positions on a single depth image. The core technique is the entanglement map, which is a feature map to measure the entanglement possibilities obtained from the input image. We use an entanglement map to select probable regions containing graspable objects. The optimum grasping pose is detected from the selected regions considering the collision between robot hand and objects. Experimental results show that our analytic method provides a more comprehensive and intuitive observation of the entanglement and exceeds previous learning-based work in success rates. Especially, our topology-based method does not rely on any object models or time-consuming training process, so that it can be easily adapted to more complex bin-picking scenes.

This repository provide the code of grasp planning by generating the entanglement map for depth images. 

## Usage

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

