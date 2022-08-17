import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

def kmeans_seg(img):
    plt.subplot(121),plt.imshow(img,'gray'),plt.title('original')
    plt.xticks([]),plt.yticks([])

    #change img(2D) to 1D
    img1 = img.reshape((img.shape[0]*img.shape[1],1))
    img1 = np.float32(img1)

    #define criteria = (type,max_iter,epsilon)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10,1.0)

    #set flags: hou to choose the initial center
    #---cv2.KMEANS_PP_CENTERS ; cv2.KMEANS_RANDOM_CENTERS
    flags = cv2.KMEANS_RANDOM_CENTERS
    flags = cv2.KMEANS_PP_CENTERS
    # apply kmenas
    compactness,labels,centers = cv2.kmeans(img1,2,None,criteria,10,flags)

    img2 = labels.reshape((img.shape[0],img.shape[1]))
    plt.subplot(122),plt.imshow(img2,'gray'),plt.title('kmeans')
    plt.xticks([]),plt.yticks([])
    plt.show()
    return img2

def canny_edge(img):
    return cv2.Canny(img, 20, 90)
root_dir = "/home/hlab/bpbot" 
img_path = os.path.join(root_dir, "data/depth/depth_cropped_pick_zone.png")
pickimg = cv2.imread(img_path,0)
img_path = os.path.join(root_dir, "data/depth/depth_cropped_pick_empty.png")
pickimg_e = cv2.imread(img_path,0)

res = canny_edge(pickimg)
res_e = canny_edge(pickimg_e)

plt.subplot(221),plt.imshow(pickimg)
plt.subplot(222),plt.imshow(res)
plt.subplot(223),plt.imshow(pickimg_e)
plt.subplot(224),plt.imshow(res_e)
plt.show()

print(f"Edge: (obj) {np.count_nonzero(res)}, (empty) {np.count_nonzero(res_e)}")
