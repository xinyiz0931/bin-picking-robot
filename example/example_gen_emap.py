import os
import cv2
import matplotlib.pyplot as plt
from bpbot.tangle_solution import LineDetection, EntanglementMap
from bpbot.utils import *
from bpbot.binpicking import *

def gen_emap():

    # tunable parameter
    len_thld = 15
    dist_thld = 3
    sliding_size = 125
    sliding_stride = 25

    img_path = os.path.join("./data", "test", "depth3.png")

    img = cv2.imread(img_path)
    norm_img = cv2.resize(adjust_grayscale(img), (250,250))

    ld = LineDetection(len_thld=len_thld,dist_thld=dist_thld)
    lines_2d, lines_3d, lines_num, drawn = ld.detect_line(norm_img, vis=True)

    # topology coordinate
    em = EntanglementMap(len_thld, dist_thld, sliding_size, sliding_stride)
    emap, wmat_vis,w,d = em.entanglement_map(norm_img)

    # control group
    lmap = em.line_map(norm_img)
    bmap = em.brightness_map(norm_img)

    # show
    fig = plt.figure()
    # fig = plt.figure()

    fig.add_subplot(241)
    plt.imshow(img, cmap='gray')
    plt.title("depth image")

    fig.add_subplot(242)
    plt.imshow(drawn)
    drawn = cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB) 
    cv2.imwrite(f"./vision/res/distance_{dist_thld}.png", drawn)
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
    plt.title("notion of sliding window")

    ax1 = fig.add_subplot(244)
    ax1.imshow(img)
    ax1.imshow(cv2.resize(emap, (img.shape[1], img.shape[0])), interpolation='bilinear', alpha=0.5, cmap='jet')
    extent = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.title("entanglement map")
    # fig.savefig(f"./vision/res/emap_size_{sliding_size}.png", bbox_inches=extent)
    # fig.savefig(f"./vision/res/emap_distance_{dist_thld}.png", bbox_inches=extent)

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
    plt.title("entanglement map (discrete)")
    
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


def detect_grasp():
    h_params = {
        "finger_height": 40,
        "finger_width":  13, 
        "open_width":    50
    }
    g_params = {
        "rotation_step": 22.5, 
        "depth_step":    50,
        "hand_depth":    50
    }
    t_params = {
        "compressed_size": 250,
        "len_thld": 15,
        "dist_thld": 3,
        "sliding_size": 125,
        "sliding_stride": 25
    }
    
    cfg = BinConfig()
    cfg = cfg.data
    # img_path = os.path.join("./data", "test", "depth4.png")

    img_path = "C:\\Users\\xinyi\\Documents\\XYBin_Pick\\bin\\tmp\\depth.png"
    img = cv2.imread(img_path)
    img_edge, num_edge = detect_edge(img, t_params)

    emap = get_entanglement_map(img, t_params)

    grasps= detect_grasp(n_grasp=10, img_path=img_path, 
                                    g_params=g_params, 
                                    h_params=h_params)
    img_grasp = draw_grasp(grasps, img.copy())

    t_grasps = detect_nontangle_grasp(n_grasp=10, img_path=img_path, 
                                    g_params=g_params, 
                                    h_params=h_params,
                                    t_params=t_params)
                                    

    t_img_grasp = draw_grasp(t_grasps, img.copy())
    # # visulization
    fig = plt.figure()

    fig.add_subplot(231)
    plt.imshow(img, cmap='gray')
    plt.title("Depth")

    fig.add_subplot(232)
    plt.imshow(img_edge)
    plt.axis("off")
    plt.title("Edge")

    fig.add_subplot(233)
    plt.imshow(img)
    plt.imshow(cv2.resize(emap, (img.shape[1], img.shape[0])), interpolation='bilinear', alpha=0.4, cmap='jet')
    plt.title("Depth + EMap")

    fig.add_subplot(234)
    plt.imshow(img_grasp)
    plt.axis("off")
    plt.title("FGE")

    fig.add_subplot(235)
    plt.imshow(t_img_grasp)
    plt.axis("off")
    plt.title("FGE + EMap")

    # grasp related
    fig.add_subplot(236)
    plt.imshow(t_img_grasp)
    plt.imshow(cv2.resize(emap, (img.shape[1], img.shape[0])), interpolation='bilinear', alpha=0.3, cmap='jet')
    plt.axis("off")
    plt.title("FGE + EMap (EMap shown)")
    
    plt.tight_layout()
    plt.get_current_fig_manager().full_screen_toggle()
    # plt.savefig("./test.png")
    plt.show()

def show_writhe_matrix():
    """Additional function for only showing edges and writhe matrix"""
    # tunable parameter
    len_thld = 15
    dist_thld = 2
    sliding_size = 125
    sliding_stride = 25
    
    img_path = os.path.join("./data", "test", "depth2.png")

    img = cv2.imread(img_path)
    norm_img = cv2.resize(adjust_grayscale(img), (250,250))
    # norm_img = cv2.medianBlur(norm_img,5)

    ld = LineDetection()
    lines_2d, lines_3d, lines_num, drawn = ld.detect_line(norm_img,len_thld, dist_thld,vis=True)
    print("line num: ", lines_num)


    em = EntanglementMap(len_thld, dist_thld,sliding_size, sliding_stride)
    # tc = TopoCoor(len_thld, dist_thld)
    emap, wmat_vis,w,d = em.entanglement_map(norm_img)

    fig = plt.figure()
    fig.add_subplot(121)
    plt.imshow(drawn)

    fig.add_subplot(122)
    plt.imshow(wmat_vis, cmap='gray')
    plt.show()
    
if __name__ == "__main__":

    import timeit
    start = timeit.default_timer()

    main()
    # show_writhe_matrix()

    end = timeit.default_timer()
    print("[*] Time: {:.2f}s".format(end - start))
