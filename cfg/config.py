# """
# Configuration class
# Arthor: Xinyi
# Datw: 2022/4/7
# """
# import os 

# class BinConfig(object):
#     # ==================== exp ====================
#     # 0 -> graspability, 1 -> pulling, 2 -> random pulling
#     exp_mode = 1

#     # ==================== image ====================
#     width = 2064
#     height = 1544

#     left_margin = 752
#     top_margin = 363
#     right_margin = 1590
#     bottom_margin = 1026
#     margins = (top_margin,left_margin,bottom_margin,right_margin)

#     max_distance = 1190
#     min_distance = 1100

#     # ==================== grasp ====================
#     # real world size
#     #finger_width = 13
#     #finger_height = 40
#     #open_width = 37
#     # slightly small gripper
#     finger_width = 10
#     finger_height = 20
#     open_width = 30
#     hand_template_size = 500
#     h_params = (finger_width, finger_width, open_width, hand_template_size)

#     rotation_step = 45
#     depth_step = 50
#     hand_depth = 50
#     g_params = (rotation_step, depth_step, hand_depth)

#     # ==================== tangle ====================
#     length_thre = 15
#     distance_thre = 3
#     sliding_size = 125
#     sliding_stride = 25
#     t_params = (length_thre, distance_thre, sliding_size, sliding_stride)
#     compressed_size = 250
#     cropped_size = 500
#     # ==================== calib ====================
#     tx = -0.474651352861443
#     ty = 0.2798040203549671
#     sx = 0.0006783468019137269
#     sy = 0.0006783468019137269
#     cz = 0.0543
#     plane_distance = 0.031

#     # ==================== force ====================
#     fx0 = 8158
#     fy0 = 8250
#     fz0 = 8546




