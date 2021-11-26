import os
import sys
import glob
sys.path.append("./")
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import math
import numpy as np
import itertools

np.set_printoptions(suppress=True)
np.seterr(divide='ignore', invalid='ignore')
import matplotlib.pyplot as plt
from scipy import ndimage
from utils.base_utils import *
from utils.image_proc_utils import *
from utils.plot_utils import *

class LineDetection(object):
    def __init__(self):
        # FLD instance, important in accuracy
        # self.length_thre = 15
        # self.distance_thre = 3
        self.canny_aperture_size = 3
        self.canny_thre1= 50
        self.canny_thre2 = 50

    def detect_line(self, src,length_thre, distance_thre, vis=False):
        """
        param src: images with 3 channels
        return:
            if only_vis: image, number
            if not: line segments 2d, 3d, number

        Attention: if source image has no lines, return None
        """
        gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)

        fld = cv2.ximgproc.createFastLineDetector(length_thre,
                                                  distance_thre,
                                                  self.canny_thre1,
                                                  self.canny_thre2,
                                                  self.canny_aperture_size)
                                                #   do_merge=False)
        lines_2d = fld.detect(gray)
        # if only_vis == True:
        #     return src_for_draw
        if lines_2d is None:
            if vis:
                return src, None
            else:
                return None, None, None
        else:
            num = lines_2d.shape[0]
            lines_reshape = lines_2d.reshape(num, 4)
            b_start = []
            b_end = []
            for each in lines_2d:
                x1, y1, x2, y2 = np.int0(each[0])
                b1 = get_max_neighbor_pixel(gray, (x1, y1))
                b2 = get_max_neighbor_pixel(gray, (x2, y2))
                b_start = np.append(b_start, b1)
                b_end = np.append(b_end, b2)
            lines_reshape = np.insert(lines_reshape, 2, b_start, axis=1)
            lines_reshape = np.insert(lines_reshape, 5, b_end, axis=1)

            if vis is True:

                src_for_draw = src.copy()
                drawn = self.draw_line_segment(src_for_draw, lines_2d, thickness=2)

                # <-------------------- revised for colored lines ---------------------->
                # src_for_draw = src.copy()
                # cmap = get_cmap(num)
                # # for i in range(14):
                # #     print(rgba2rgb(cmap(i)))
                # for i in range(num):
                #     each = lines_2d[i]
                #     x1, y1, x2, y2 = np.int0(each[0])
                #     color = rgba2rgb(cmap(i))
                    # drawn = cv2.line(src_for_draw, (x1, y1), (x2, y2), color, thickness=2)
                
                # <-------------------- revised for colored lines ---------------------->
                
                return lines_2d, lines_reshape.reshape((num, 1, 6)), num, drawn
            else:
                return lines_2d, lines_reshape.reshape((num, 1, 6)), num

    def draw_line_segment(self, _image, lines_2d, color=(255, 0, 0), thickness=1):
        """_image is just for draw, not source image"""
        for each in lines_2d:
            x1, y1, x2, y2 = np.int0(each[0])
            _image = cv2.line(_image, (x1, y1), (x2, y2), color, thickness)
        return _image
            
class TopoCoor(object):
    def __init__(self, length_thre=15, distance_thre=3):
        self.length_thre = length_thre
        self.distance_thre = distance_thre

    def cross_product(self, a, b):
        x1, y1, z1 = a
        x2, y2, z2 = b
        return np.array([(y1 * z2 - y2 * z1), -(x1 * z2 - x2 * z1), (x1 * y2 - x2 * y1)])

    def gli_original(self, line1, line2):
        """
        Calculate Gaussian link integral in a simple way
        Input: two line segment
        output:
        """
        a = np.array([line1[0], line1[1], line1[2]])
        b = np.array([line1[3], line1[4], line1[5]])

        c = np.array([line2[0], line2[1], line2[2]])
        d = np.array([line2[3], line2[4], line2[5]])

        r_ac = c - a
        r_ad = d - a
        r_bc = c - b
        r_bd = d - b

        c_ac_ad = self.cross_product(r_ac, r_ad)
        c_ad_bd = self.cross_product(r_ad, r_bd)
        c_bd_bc = self.cross_product(r_bd, r_bc)
        c_bc_ac = self.cross_product(r_bc, r_ac)

        n_a = c_ac_ad / np.linalg.norm(c_ac_ad)
        n_b = c_ad_bd / np.linalg.norm(c_ad_bd)
        n_c = c_bd_bc / np.linalg.norm(c_bd_bc)
        n_d = c_bc_ac / np.linalg.norm(c_bc_ac)

        w = np.arcsin(np.dot(n_a, n_b)) + np.arcsin(np.dot(n_b, n_c)) + np.arcsin(np.dot(n_c, n_d)) + np.arcsin(
            np.dot(n_d, n_a))
        return np.nan_to_num(w)
    
    def topo_coor_from_edge(self, lines_3d):
        """Given lines segments from edges, output the topology coordinates
        Args:
            lines ([array]): shape=(num, 1, 6))
        Return:
            writhe_matrix
            (writhe, density, center)
        """
        lines_num = lines_3d.shape[0]
        writhe_matrix = np.zeros([lines_num, lines_num])
        wm_flatten = np.array([])
        for i in range(lines_num):
            for j in range(i + 1, lines_num):
                p0, p1, p2, p3, p4, p5 = lines_3d[i][0]
                q0, q1, q2, q3, q4, q5 = lines_3d[j][0]
                l1 = [p0, p1, p2, p3, p4, p5]
                l2 = [q0, q1, q2, q3, q4, q5]
                writhe = self.gli_original(l1, l2)
                writhe_matrix[i][j] = writhe
                wm_flatten = np.append(wm_flatten, writhe)

        # start computing writhe(avg.)
        writhe = np.sum(writhe_matrix) # we try to use the total writhe of the matrix
        # start computing density
        density_thre = np.mean(wm_flatten)
        if len(wm_flatten):
            density = len(wm_flatten[wm_flatten >= density_thre]) / len(wm_flatten)
        else:
            density = 0
        
        return writhe_matrix, writhe, density

    def topo_coor_from_two_edges(self, line1, line2):
        """Given lines segments from edges, output the topology coordinates
        Args:
            lines ([array]): shape=(num, 1, 6))
        Return:
            writhe_matrix
            (writhe, density, center)
        """
        lnum1 = line1.shape[0]
        lnum2 = line2.shape[0]
        writhe_matrix = np.zeros([lnum1, lnum2])
        wm_flatten = np.array([])

        for i in range(lnum1):
            for j in range(lnum2):
                p0, p1, p2, p3, p4, p5 = line1[i][0]
                q0, q1, q2, q3, q4, q5 = line2[j][0]
                l1 = [p0, p1, p2, p3, p4, p5]
                l2 = [q0, q1, q2, q3, q4, q5]
                writhe = self.gli_original(l1, l2)
                writhe_matrix[i][j] = writhe
                wm_flatten = np.append(wm_flatten, writhe)
        writhe = np.sum(writhe_matrix) # we try to use the total writhe of the matrix
        # start computing density
        density_thre = np.mean(wm_flatten)
        if len(wm_flatten):
            density = len(wm_flatten[wm_flatten >= density_thre]) / len(wm_flatten)
        else:
            density = 0
        
        return writhe_matrix, writhe, density

    def topo_coor_from_image(self, src, cmask=False):
        """Given an image, output the topology coordinates
        Args:
            src ([image]): 3-channel image
        Return:
            writhe_matrix
            (writhe, density, center)
            if_center_mask is True: return a mask same size as src. for overall analysis
            if_center_maks is False: no mask return. for emap generation
        """
        ld = LineDetection()
        lines_2d, lines_3d, lines_num = ld.detect_line(src, self.length_thre, self.distance_thre)
        if lines_num is None:
            if cmask is False:
                return None, None ,None
            else:
                return None, None ,None, None

        else:
            writhe_matrix = np.zeros([lines_num, lines_num])
            wm_flatten = np.array([])
            for i in range(lines_num):
                for j in range(i + 1, lines_num):
                    p0, p1, p2, p3, p4, p5 = lines_3d[i][0]
                    q0, q1, q2, q3, q4, q5 = lines_3d[j][0]
                    l1 = [p0, p1, p2, p3, p4, p5]
                    l2 = [q0, q1, q2, q3, q4, q5]
                    writhe = self.gli_original(l1, l2)
                    writhe_matrix[i][j] = writhe
                    wm_flatten = np.append(wm_flatten, writhe)
            
            # start computing writhe(avg.)
            writhe = np.sum(writhe_matrix) / lines_num  # here we use the average of writhe matrix

            # start computing density
            DENSITY_THRE = np.mean(wm_flatten)
            if len(wm_flatten):
                density = len(wm_flatten[wm_flatten >= DENSITY_THRE]) / len(wm_flatten)
            else:
                density = 0
            
            if cmask is False:
               
                return writhe_matrix, writhe, density
            else:
                # start computing center
                (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(writhe_matrix)
                center = maxLoc
                # center = np.int0(ndimage.center_of_mass(writhe_matrix))

                # generate center mask
                c1 = lines_2d[center[0]][0]
                c2 = lines_2d[center[1]][0]
                center_mask = np.zeros((src.shape[0], src.shape[1]))
                # (x1,y1): left-top, (x2,y2):right-bottom
                x1 = int(np.min([c1[0], c1[2], c2[0], c2[2]]))
                x2 = int(np.max([c1[0], c1[2], c2[0], c2[2]]))
                y1 = int(np.min([c1[1], c1[3], c2[1], c2[3]]))
                y2 = int(np.max([c1[1], c1[3], c2[1], c2[3]]))
                # draw center for test
                # test = cv2.line(src, (int(c1[0]), int(c1[1])), (int(c1[2]), int(c1[3])), (0, 255.0), thickness=3)
                # test = cv2.line(test, (int(c2[0]), int(c2[1])), (int(c2[2]), int(c2[3])), (0, 255, 0), thickness=3)
                center_mask[y1:y2, x1:x2] = 255
                # cv2.imwrite("./DEPTH_TMP/center_line.png", test)
                # cv2.imwrite("./DEPTH_TMP/center_mask.png", center_mask)
                return writhe_matrix, writhe, density, center_mask

    def compute_writhe_matrix(graph):
        """graph: shape=(2(only 2 objs) x M(template nodes) x 3)"""
        graph = np.array(graph)
        obj_collection = []
        for nodes in graph:
            # obj number
            obj_collection.append(node2edge(nodes))

        obj_collection = np.array(obj_collection)

        n_seg = obj_collection.shape[1]

        gli_mat = np.zeros((n_seg, n_seg))
        obj1 = obj_collection[0]
        obj2 = obj_collection[1]
        for i in range(n_seg):
            for j in range(n_seg):
                gli = gli_original(obj1[i], obj2[j])
                gli_mat[i][j] = gli
        return gli_mat

def main():

    # tunable parameter
    # length_thre = 15
    # distance_thre = 3
    length_thre = 15
    distance_thre = 3
    sliding_size = 125
    sliding_stride = 25

    img_path = "./vision/depth/depth0.png"
    # img_path = "D:\\code\\myrobot\\vision\\test\\mask3.png"

    img = cv2.imread(img_path)
    norm_img = cv2.resize(adjust_grayscale(img), (250,250))

    ld = LineDetection()
    lines_2d, lines_3d, lines_num, drawn = ld.detect_line(norm_img,length_thre, distance_thre,vis=True)

    # topology coordinate
    tc = TopoCoor(length_thre, distance_thre,sliding_size, sliding_stride)
    emap, wmat_vis,w,d = tc.entanglement_map(norm_img)

    # control group #1
    lmap = tc.line_map(norm_img)
    bmap = tc.brightness_map(norm_img)

    # show
    fig = plt.figure()
    # fig = plt.figure()

    fig.add_subplot(241)
    plt.imshow(img, cmap='gray')
    plt.title("depth image")

    fig.add_subplot(242)
    plt.imshow(drawn)
    drawn = cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB) 
    cv2.imwrite(f"./vision/res/distance_{distance_thre}.png", drawn)
    plt.title("edges")

    # fig.add_subplot(243)
    # window_example = cv2.rectangle(norm_img.copy(), (20,20),(20+sliding_size, 20+sliding_size), (0,255,0), 2)
    # cv2.imwrite(f"./vision/res/window_size_{sliding_size}.png", window_example)
    # plt.imshow(window_example)

    fig.add_subplot(243)
    window_example = cv2.rectangle(norm_img.copy(), (20,20),(20+sliding_size, 20+sliding_size), (0,175,0), 2)
    window_example = cv2.rectangle(window_example.copy(), (20+sliding_stride,20+sliding_stride),(20+sliding_size+sliding_stride, 20+sliding_size+sliding_stride), (0,255,0), 2)
    # cv2.imwrite(f"./vision/res/window_stride_{sliding_stride}.png", window_example)
    plt.imshow(window_example)

    ax1 = fig.add_subplot(244)
    ax1.imshow(img)
    ax1.imshow(cv2.resize(emap, (img.shape[1], img.shape[0])), interpolation='bilinear', alpha=0.5, cmap='jet')
    extent = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    # fig.savefig(f"./vision/res/emap_size_{sliding_size}.png", bbox_inches=extent)
    fig.savefig(f"./vision/res/emap_distance_{distance_thre}.png", bbox_inches=extent)


    fig.add_subplot(245)
    plt.imshow(lmap, cmap='jet')
    for i in range(lmap.shape[1]):
        for j in range(lmap.shape[0]):
            text = plt.text(j, i, int(lmap[i,j]), ha="center", va="center", color="w")
    plt.title("line map")
    
    fig.add_subplot(246)
    plt.imshow(bmap, cmap='jet')
    for i in range(bmap.shape[1]):
        for j in range(bmap.shape[0]):
            text = plt.text(j, i, np.round(bmap[i, j],2), ha="center", va="center", color="w")
    plt.title("brightness map")
    
    fig.add_subplot(247)
    plt.imshow(emap, cmap='jet')
    for i in range(emap.shape[1]):
        for j in range(emap.shape[0]):
            text = plt.text(j, i, np.round(emap[i, j],2),ha="center", va="center", color="w")

    # plt.imshow(img, alpha=0.2,cmap='gray')
    
    plt.title("entanglement map")
    
    plt.tight_layout()
    plt.get_current_fig_manager().full_screen_toggle()
    plt.show()

    # just to save the image
    # plt.imshow(img)
    # plt.imshow(cv2.resize(emap, (img.shape[1], img.shape[0])), interpolation='bilinear', alpha=0.5, cmap='jet')
    # plt.savefig('./grasp/edmap.png', dpi=600, bbox_inches='tight')
    # plt.show()

    # plt.imshow(cv2.resize(emap, (img.shape[1], img.shape[0])), interpolation='bilinear', cmap='jet')
    # plt.savefig('./grasp/emap.png', dpi=600, bbox_inches='tight')
    # plt.show()




if __name__ == "__main__":

    import timeit
    start = timeit.default_timer()
    length_thre = 15
    distance_thre = 2
    sliding_size = 125
    sliding_stride = 25
    
    # main()
    img_path = "./vision\\test\\depth3.png"

    img = cv2.imread(img_path)
    norm_img = cv2.resize(adjust_grayscale(img), (250,250))
    # norm_img = cv2.medianBlur(norm_img,5)

    ld = LineDetection()
    lines_2d, lines_3d, lines_num, drawn = ld.detect_line(norm_img,length_thre, distance_thre,vis=True)
    print("line num: ", lines_num)

    # topology coordinate
    tc = TopoCoor(length_thre, distance_thre,sliding_size, sliding_stride)
    emap, wmat_vis,w,d = tc.entanglement_map(norm_img)

    fig = plt.figure()
    fig.add_subplot(121)
    plt.imshow(drawn)

    fig.add_subplot(122)
    plt.imshow(wmat_vis, cmap='gray')
    plt.show()

    end = timeit.default_timer()
    main_proc_print("Time: {:.2f}s".format(end - start))
