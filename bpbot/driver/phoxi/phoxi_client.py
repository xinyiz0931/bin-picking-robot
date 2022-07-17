import grp
import sys

from torch import repeat_interleave
from os import path
import grpc
import numpy as np
import phoxi_pb2 as pxmsg
import phoxi_pb2_grpc as pxrpc
import copy

class PhxClient(object):

    def __init__(self, host = "localhost:18300"):
        options = [('grpc.max_receive_message_length', 100 * 1024 * 1024)]
        channel = grpc.insecure_channel(host, options=options)
        self.stub = pxrpc.PhoxiStub(channel)

    def __unpackarraydata(self, dobj):
        h = dobj.width
        w = dobj.height
        ch = dobj.channel
        return copy.deepcopy(np.frombuffer(dobj.image).reshape((w,h,ch)))

    def triggerframe(self):
        try: 
            self.stub.triggerframe(pxmsg.Null())
        except grpc.RpcError as rpc_error:
            print(f'Failed with {rpc_error.code()}')
    
    def saveplyauto(self):
        """
        auto saveply at `/tmp/out.ply`
        author: xinyi
        date: 20210726
        """
        try:
            self.stub.saveplyauto(pxmsg.Null())
        except grpc.RpcError as rpc_error:
           print(f'Failed with {rpc_error.code()}') 
    
    def saveply(self, save_dir):
        """
        saveply at some location
        author: xinyi
        date: 20210726
        """
        try:
            self.stub.saveply(pxmsg.SaveDir(path=save_dir))
        except grpc.RpcError as rpc_error:
            print(f'Failed with {rpc_error.code()}')

    def gettextureimg(self):
        """
        get gray image as an array

        :return: a textureHeight*textureWidth*1 np array
        author: weiwei
        date: 20191202
        """

        txtreimg = self.stub.gettextureimg(pxmsg.Null())
        txtrenparray = self.__unpackarraydata(txtreimg)
        
        maxvalue = np.amax(txtrenparray)
        print(maxvalue)
        txtrenparray = txtrenparray/maxvalue*255
        return txtrenparray.astype(np.uint8)

    def getgrayscaleimg(self):
        try:
            gsimg = self.stub.gettextureimg(pxmsg.Null())
            gsarray = self.__unpackarraydata(gsimg)
            gsarray[gsarray > 255] = 255
            return gsarray.astype(np.uint8)
        except grpc.RpcError as rpc_error:
            print(f"Failed with {rpc_error.code()}")
            return None

    def getdepthimg(self):
        """
        get depth image as an array

        :return: a depth array array (float32)

        author: weiwei
        date: 20191202
        """

        depthimg = self.stub.getdepthimg(pxmsg.Null())
        depthnparray = self.__unpackarraydata(depthimg)
        depthnparray_float32 = copy.deepcopy(depthnparray)
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depthnparray, alpha=0.08), cv2.COLORMAP_JET)
        # convert float32 deptharray to unit8
        # maxdepth = np.max(depthnparray)
        # mindepth = np.min(depthnparray[depthnparray != 0])
        # depthnparray[depthnparray!=0] = (depthnparray[depthnparray!=0]-mindepth)/maxdepth*200+55
        # depthnparray = depthnparray.astype(dtype= np.uint8)
        # depthnparray_scaled = copy.deepcopy(depthnparray)
        return depthnparray_float32

    def getpcd(self):
        """
        get the full poind cloud of a new frame as an array

        :return: np.array point cloud n-by-3
        author: weiwei
        date: 20191202
        """
        try:
            pcd = self.stub.getpcd(pxmsg.Null())
            return np.frombuffer(pcd.points).reshape((-1,3))
        except grpc.RpcError as rpc_error:
            print(f"Failed with {rpc_error.code()}")
            return None 

    def getnormals(self):
        """
        get the the normals of the pointcloudas an array

        :return: np.array n-by-3
        author: weiwei
        date: 20191208
        """

        nrmls = self.stub.getnormals(pxmsg.Null())
        return np.frombuffer(nrmls.points).reshape((-1,3))

    def cvtdepth(self, darr_float32):
        """
        convert float32 deptharray to unit8

        :param darr_float32:
        :return:

        author: weiwei
        date: 20191228
        """

        depthnparray_scaled = copy.deepcopy(darr_float32)
        maxdepth = np.max(darr_float32)
        mindepth = np.min(darr_float32[darr_float32 != 0])
        depthnparray_scaled[depthnparray_scaled!=0] = (darr_float32[darr_float32!=0]-mindepth)/(maxdepth-mindepth)*200+25
        depthnparray_scaled = depthnparray_scaled.astype(dtype= np.uint8)

        return depthnparray_scaled

if __name__ == "__main__":

    import timeit
    start = timeit.default_timer()

    import phoxi_client as pclt
    pxc = pclt.PhxClient(host ="127.0.0.1:18300")

    pxc.triggerframe()
    # pcd = pxc.getpcd()
    # img = pxc.getgrayscaleimg()
    # print(img.max())
    import cv2
    #cv2.imshow("w", cv2.resize(img, [640,480]))
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    end = timeit.default_timer()
    print("Time cost: ", end-start)
    # pxc.saveply("../../vision/pointcloud/out.ply")
    # print("point cloud is captured and saved! ")
