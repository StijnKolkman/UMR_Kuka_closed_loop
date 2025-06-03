import socket  # Used for TCP/IP communication
import time
import threading


class KUKA(socket.socket):
    def ping_loop(self):
        while not self.ping_stop.wait(5):
            if self.read("PING") != "PONG":
                print("Ping failed. Disconnecting...")
                self.disconnect()

    def __init__(self, TCP_IP):
        super().__init__(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.settimeout(10)
            self.connect(
                (TCP_IP, 7000)
            )  # Open socket. kukavarproxy actively listens on TCP port 7000
            self.input_stream = self.makefile("rb")
            self.output_stream = self.makefile("wb")
            self.ping_stop = threading.Event()
            self.x = threading.Thread(target=self.ping_loop, daemon=True)
            self.lock = threading.Lock()
            self.x.start()
        except socket.error as e:
            print(f"Socket error: {e}")
            print("Failed to connect to KUKA. Check the IP address and kukavarproxy.")

    def disconnect(self):
        self.ping_stop.set()
        self.x.join()
        return

    def get_response(self) -> str:
        return self.decode_kuka(self.recv(1024))

    def send_msg(self, var, rw, val, msgID):
        """
        kukavarproxy message format is
        msg ID in HEX                       2 bytes
        msg length in HEX                   2 bytes
        read (0) or write (1)               1 byte
        variable name length in HEX         2 bytes
        variable name in ASCII              # bytes
        variable value length in HEX        2 bytes
        variable value in ASCII             # bytes
        """
        try:
            msg = bytearray()
            temp = bytearray()
            if val != "":
                val = str(val)
                msg.append((len(val) & 0xFF00) >> 8)  # MSB of variable value length
                msg.append((len(val) & 0x00FF))  # LSB of variable value length
                msg.extend(map(ord, val))  # Variable value in ASCII
            temp.append(bool(rw))  # Read (0) or Write (1)
            temp.append(((len(var)) & 0xFF00) >> 8)  # MSB of variable name length
            temp.append((len(var)) & 0x00FF)  # LSB of variable name length
            temp.extend(map(ord, var))  # Variable name in ASCII
            msg = temp + msg
            del temp[:]
            temp.append((msgID & 0xFF00) >> 8)  # MSB of message ID
            temp.append(msgID & 0x00FF)  # LSB of message ID
            temp.append((len(msg) & 0xFF00) >> 8)  # MSB of message length
            temp.append((len(msg) & 0x00FF))  # LSB of message length
            msg = temp + msg
        except:
            self.error_list(2)
        self.send(msg)
        return self.get_response()

    def decode_kuka(self, res):
        rw = res[4]
        var_name_len = int.from_bytes(res[5:7], byteorder="big")
        val = res[7 : 7 + var_name_len].decode("ascii")
        return val

    def read(self, var, msgID=0):
        with self.lock:
            res = self.send_msg(var, 0, "", msgID)
        return res

    def write(self, var, val, msgID=0):
        with self.lock:
            try:
                if val != (""):
                    res = self.send_msg(var, 1, val, msgID)
                else:
                    raise self.error_list(3)
            except:
                self.error_list(2)
        return res

    def error_list(self, ID):
        if ID == 1:
            print("Network Error (tcp_error)")
            print(
                "    Check your KRC's IP address on the network, and make sure kukaproxyvar is running."
            )
            self.close()
            raise SystemExit
        elif ID == 2:
            print("Python Error.")
            print(
                "    Check the code and uncomment the lines related to your python version."
            )
            self.close()
            raise SystemExit
        elif ID == 3:
            print("Error in write() statement.")
            print("    Variable value is not defined.")
