import grpc
import dynpick_pb2 as dpmsg
import dynpick_pb2_grpc as dprpc
import numpy as np
np.set_printoptions(suppress=True)

class DynPickClient(object):
    def __init__(self):
        try: 
            options = [('grpc.max_receive_message_length', 100 * 1024 * 1024)]
            channel = grpc.insecure_channel('localhost:2222', options=options)
            self.stub = dprpc.DynPickStub(channel)
        except grpc.RpcError as rpc_error: 
            print(f"[!] Failed to connect DynPickServer: {rpc_error.code()}")
    
    def get(self):
        # get raw data: int type
        buffer = self.stub.get(dpmsg.Nothing())
        return np.frombuffer(buffer.data, dtype=int)
    
    def save(self):
        self.stub.save(dpmsg.Nothing())
        return
    def save_ok(self): 
        self.stub.save_ok(dpmsg.Nothing())
        return


if __name__ == '__main__':
    import time
    dynpick_client = DynPickClient()
    # while True:
    #     print(dynpick_client.get()[2])
    import timeit
    start_time = timeit.default_timer()
    print("save start")
    print(dynpick_client.get())
    # dynpick_client.save()
    # time.sleep(3)
    # print("save stop")
    # dynpick_client.save_ok()
    print("total: ", timeit.default_timer() - start_time) 
