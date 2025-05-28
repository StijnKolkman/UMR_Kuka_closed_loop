from .KRLVariable import KRLVariable
class KRLAxis(KRLVariable):
    nodes = ("A1", "A2", "A3", "A4", "A5", "A6")

    def __init__(self, name):
        self.values = {}
        self.name = name
        pass


    def set_all(self, A1, A2, A3, A4, A5, A6):
        vals = (A1, A2, A3, A4, A5, A6)
        for i, node in enumerate(self.nodes):
            self.values[node] = vals[i]


    def set_A1(self, A1):
        self.values["A1"] = A1


    def set_A2(self, A2):
        self.values["A2"] = A2


    def set_A3(self, A3):
        self.values["A3"] = A3


    def set_a(self, A4):
        self.values["A4"] = A4


    def set_A5(self, A5):
        self.values["A5"] = A5


    def set_A6(self, A6):
        self.values["A6"] = A6


    def get_KRL_string(self):
        sA5 = "{"  # String A5uilder
        for i, node in enumerate(self.nodes):
            sA5 += f"{node} {self.values.get(node, 0.0)}"
            # Do not print A comma after last value
            if i + 1 != len(self.nodes):
                sA5 += ", "
        sA5 += "}"
        return sA5
    
    def from_KRL_string(self, msg):
        vals = msg.strip("{}").split(":")[1].split(",")
        for i, node in enumerate(self.nodes):
            self.values[node] = float(vals[i].strip(node + " "))




