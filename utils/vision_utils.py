import cv2
import matplotlib.pyplot as plt
import numpy as np
# import open3d as o3d
def rotate_img(img, angle, center=None, scale=1.0):
    (h,w) = img.shape[:2]

    if center is None:
        center=(w/2, h/2)

    M = cv2.getRotationMatrix2D(center, angle,scale)
    rotated = cv2.warpAffine(img, M, (w,h))
    return rotated

def rotate_3d(p, R, origin=(0, 0, 0)):
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T - o.T) + o.T).T)
    
def rotate_point(img, loc, angle, center=None, scale=1.0):
    (h,w) = img.shape[:2]
    x,y, = loc
    if center is None:
        center=(w/2, h/2)

    M = cv2.getRotationMatrix2D(center, angle,scale)
    rotated = cv2.warpAffine(img, M, (w,h))
    rot_loc = np.dot(M,[x,y,1])
    return rot_loc

def get_neighbor_bounding(gray, loc, bounding_size=10, bounding_draw=False):
    """Give a mat and a location, return the max neighobor pixel value
       Xinyi

    Arguments:
        gray {grayscale or array} -- images with 1 channels, shape = (h,w)
        loc {(int, int)} -- location, (left,top), (from width,from height) 

    Keyword Arguments:
        bounding_size {int} -- [neighbor bounding size] (default: {10})
        bounding_draw {bool} -- [if draw rectangle to verify position] (default: {False})

    Returns:
        [int] -- max value of neighbor pixels
    """
    (x,y)=loc
    h, w = gray.shape

    # construct the refined matrix, first row then column
    r_con = np.zeros((bounding_size,w),dtype=np.uint8)
    mat_tmp = np.r_[np.r_[r_con,gray],r_con]
    c_con = np.zeros((h+bounding_size*2,bounding_size),dtype=np.uint8)
    mat_tmp = np.c_[np.c_[c_con,mat_tmp],c_con]

    if bounding_draw:
        # draw the bounding box to verify the correct pixel
        for_draw = cv2.cvtColor(gray.copy(), cv2.COLOR_GRAY2RGB) # copy the image for draw
        cv2.circle(for_draw, (x,y), 6, (255,0,0), -1)
        cv2.rectangle(for_draw, ((x-bounding_size),(y-bounding_size)),((x+bounding_size+1),(y+bounding_size+1)), (255,0,0),2)
        plt.figure(figsize=(15,15)) 
        plt.imshow(for_draw, cmap='gray'), plt.show()

    x = x+bounding_size
    y = y+bounding_size
    return mat_tmp[(y-bounding_size):(y+bounding_size+1),(x-bounding_size):(x+bounding_size+1)]


def get_max_neighbor_pixel(gray, loc, bounding_size=10, bounding_draw=False):
    """Give a mat and a location, return the max neighobor pixel value
       Xinyi

    Arguments:
        gray {grayscale or array} -- images with 1 channels, shape = (h,w)
        loc {(int, int)} -- location, (left,top), (from width,from height) 

    Keyword Arguments:
        bounding_size {int} -- [neighbor bounding size] (default: {10})
        bounding_draw {bool} -- [if draw rectangle to verify position] (default: {False})

    Returns:
        [int] -- max value of neighbor pixels
    """
    (x,y)=loc
    h, w = gray.shape

    # construct the refined matrix, first row then column
    r_con = np.zeros((bounding_size,w),dtype=np.uint8)
    mat_tmp = np.r_[np.r_[r_con,gray],r_con]
    c_con = np.zeros((h+bounding_size*2,bounding_size),dtype=np.uint8)
    mat_tmp = np.c_[np.c_[c_con,mat_tmp],c_con]

    if bounding_draw:
        # draw the bounding box to verify the correct pixel
        for_draw = cv2.cvtColor(gray.copy(), cv2.COLOR_GRAY2RGB) # copy the image for draw
        cv2.circle(for_draw, (x,y), 6, (255,0,0), -1)
        cv2.rectangle(for_draw, ((x-bounding_size),(y-bounding_size)),((x+bounding_size+1),(y+bounding_size+1)), (255,0,0),2)
        plt.figure(figsize=(15,15)) 
        plt.imshow(for_draw, cmap='gray'), plt.show()

    x = x+bounding_size
    y = y+bounding_size
    return np.max(mat_tmp[(y-bounding_size):(y+bounding_size+1),(x-bounding_size):(x+bounding_size+1)])

def get_neighbor_pixel(gray,loc,bounding_size=10):
    (x,y) = loc
    h,w = gray.shape
    mat = gray[(y-bounding_size):(y+bounding_size+1),(x-bounding_size):(x+bounding_size+1)]
    left_margin = x-bounding_size
    top_margin = y-bounding_size

    # index = w * Y + X
    max_xy = np.where(mat == mat.max())

    y_p = max_xy[0][0] + y-bounding_size
    x_p = max_xy[1][0] + x-bounding_size
    return (x_p, y_p),mat.max()


def normalize_depth_map(gray_array, max_distance, min_distance, img_width, img_height):
    """normalize an array to range[0,255], display(optional)

    Arguments:
        gray_array {numpy array} -- 1channel, with camera z-value, unit: mm
        max_distance {int} -- camera max distance
        min_distance {int} -- camera min distance
        img_width {int} -- depth image width, need to be the same as camera
        img_height {int} -- depth image height, need to be the same as camera

    Returns:
        grayscale image -- depth map
    """

    gray_array[gray_array > max_distance] = max_distance
    gray_array[gray_array < min_distance] = min_distance
    # gray_array = - gray_array
    img = ((gray_array - min_distance) * (1/(max_distance - min_distance) * 255)).astype('uint8')
    img = (-img).reshape((img_height, img_width))

    img[img==255]=80
    # remove pixels of 255
    
    # cv2.imshow("windows", img)    
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return img

def adjust_grayscale(img, max_b=255):
        
    # adjust depth maps
    img_adjusted = np.array(img * (max_b/np.max(img)), dtype=np.uint8)
    return img_adjusted


def adjust_array_range(array, range=(0,1), if_img=False):
    # adjust depth maps
    if if_img == False:
        return np.interp(array, (array.min(), array.max()), range)
    else:
        return np.array(np.interp(array, (array.min(), array.max()), range), dtype=np.uint8)

def depth2pc(gray, pc_path=None):
    h, w = gray.shape
    y_ = np.linspace(1, h, h)
    x_ = np.linspace(1, w, w)
    mesh_x, mesh_y = np.meshgrid(x_,y_)
    z_ = gray.flatten()

    xyz = np.zeros((np.size(mesh_x), 3))

    xyz[:, 0] = np.reshape(mesh_x, -1)
    xyz[:, 1] = np.reshape(mesh_y, -1)
    xyz[:, 2] = np.reshape(z_, -1)
    xyz = np.delete(xyz, np.where(xyz[:, 2]==0)[0], axis=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.visualization.draw_geometries([pcd])

    if pc_path is not None:
        o3d.io.write_point_cloud(pc_path, pcd)

    # o3d.io.read_point_cloud(pc_path)
    # o3d.visualization.draw_geometries([pcd])
    return pcd


def depth2xyz(gray):
    h, w = gray.shape
    y_ = np.linspace(1, h, h)
    x_ = np.linspace(1, w, w)
    mesh_x, mesh_y = np.meshgrid(x_,y_)
    z_ = gray.flatten()

    xyz = np.zeros((np.size(mesh_x), 3))

    xyz[:, 0] = np.reshape(mesh_x, -1)
    xyz[:, 1] = np.reshape(mesh_y, -1)
    xyz[:, 2] = np.reshape(z_, -1)
    xyz = np.delete(xyz, np.where(xyz[:, 2]==0)[0], axis=0)
    return xyz

def xyz2depth(gray_array, max_distance, min_distance, img_width, img_height):
    """normalize an array to range[0,255], display(optional)

    Arguments:
        gray_array {numpy array} -- 1channel, with camera z-value, unit: mm
        max_distance {int} -- camera max distance
        min_distance {int} -- camera min distance
        img_width {int} -- depth image width, need to be the same as camera
        img_height {int} -- depth image height, need to be the same as camera

    Returns:
        grayscale image -- depth map
    """

    gray_array[gray_array > max_distance] = max_distance
    gray_array[gray_array < min_distance] = min_distance
    # gray_array = - gray_array
    img = ((gray_array - min_distance) * (1 / (max_distance - min_distance) * 255)).astype('uint8')
    img = (-img).reshape((img_height, img_width))

    img_adjusted = np.array(img * (255/np.max(img)), dtype=np.uint8)

    # remove pixels of 255

    # cv2.imshow("windows", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return img_adjusted

def process_raw_pc(pcd_path):
    pcd = o3d.io.read_point_cloud(pcd_path)
    # o3d.visualization.draw_geometries([pcd])

    xyz = np.asarray(pcd.points)
    xyz = np.delete(xyz, np.where(xyz[:, 1] <= 1)[0], axis=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    pc_path = "./vision/tmp/reform.ply"
    o3d.io.write_point_cloud(pc_path, pcd)
    re_pcd = o3d.io.read_point_cloud(pc_path)
    down_pcd = re_pcd.voxel_down_sample(voxel_size=8)

    re_xyz = np.asarray(down_pcd.points)
    return (re_xyz)


def process_raw_xyz(xyz_path):
    xyz = np.loadtxt(xyz_path)
    xyz = np.delete(xyz, np.where(xyz[:, 1] <= 1)[0], axis=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    pc_path = "./vision/tmp/reform.ply"
    o3d.io.write_point_cloud(pc_path, pcd)
    re_pcd = o3d.io.read_point_cloud(pc_path)
    down_pcd = re_pcd.voxel_down_sample(voxel_size=8)

    re_xyz = np.asarray(down_pcd.points)
    return (re_xyz)

def reform_xyz(xyz):
    xyz = np.delete(xyz, np.where(xyz[:, 1] <= 1)[0], axis=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    pc_path = "./vision/tmp/reform.ply"
    o3d.io.write_point_cloud(pc_path, pcd)
    re_pcd = o3d.io.read_point_cloud(pc_path)
    down_pcd = re_pcd.voxel_down_sample(voxel_size=8)

    re_xyz = np.asarray(down_pcd.points)
    return (re_xyz)



if __name__ == "__main__":

    """Varification code here. """
    # img = cv2.imread("./vision/depth/depthc.png",0)
    # plt.imshow(img)
    # cc = adjust_grayscale(img)
    # plt.imshow(cc)
    # plt.show()
    a = np.array([[1,2,3]])
    b= adjust_array_range(a,range=(0,0.5))
    print(b)
    # img = cv2.imread("../vision/test/mask3.png", 0)
    # adjusted = adjust_grayscale(img, max_b=100)
    # plt.imshow(adjusted)
    # plt.show()
    # x,y = [1168,656]

    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # o = gray[x,y]
    # # r = get_max_neighbor_pixel(gray, [x,y], bounding_draw=True)
    # r = get_max_neighbor_pixel(gray, [x,y])

    # print(f"Original value from image is {o}, real value is {r}.")

    # print(get_neighbor_pixel(gray,[x,y], bounding_size=3))

