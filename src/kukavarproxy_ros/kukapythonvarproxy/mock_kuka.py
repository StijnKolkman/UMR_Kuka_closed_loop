import time
import threading
import inspect

from .KRL_Pos import KRLPos
from .KRL_Axis import KRLAxis

class fakeKUKA():

    def __init__(self):
        self.lock = threading.Lock()
        self.last_zero = time.time() - 1
        pass

    def disconnect(self):
        self.running = False
        return

    def get_response(self) -> str:
        return self.decode_kuka(self.recv(1024))


    def send_msg(self, var, rw,val,msgID):
        return 0
    def decode_kuka(self, res):
        rw = res[4]
        var_name_len = int.from_bytes(res[5:7])
        val = res[7:7 + var_name_len].decode('ascii')
        return val

    def read (self, var, msgID=0):
        if var == "COM_ACTION":
            current_time = time.time()
            if current_time - self.last_zero >= 1:
                self.last_zero = current_time
                return "0"
            return "1"
        caller_frame = inspect.currentframe().f_back  # Get the previous frame
        caller_self = caller_frame.f_locals.get("self")  # Try to get `self`
        res = "0"
        if caller_self and hasattr(caller_self, "get_KRL_string"):
            res= caller_self.get_KRL_string()
        if isinstance(caller_self, KRLPos) or isinstance(caller_self, KRLAxis) :
            res = "{ :" + res.strip("{}") + "}" 
        return res


    def write (self, var, val, msgID=0):
        if var == "COM_ACTION":
            if val != 15:
                self.last_zero = time.time() - 1
        with self.lock:
            try:
                if val != (""): 
                    res = self.send_msg(var,1, val,msgID)
                else: 
                    raise self.error_list(3)
            except Exception as e:
                print(e)
                self.error_list(2)
        return res

    def error_list (self, ID):
        if ID == 1:
            print ("Network Error (tcp_error)")
            print ("    Check your KRC's IP address on the network, and make sure kukaproxyvar is running.")
            self.close()
            raise SystemExit
        elif ID == 2:
            print ("Python Error.")
            print ("    Check the code and uncomment the lines related to your python version.")
            self.close()
            raise SystemExit
        elif ID == 3:
            print ("Error in write() statement.")
            print ("    Variable value is not defined.")