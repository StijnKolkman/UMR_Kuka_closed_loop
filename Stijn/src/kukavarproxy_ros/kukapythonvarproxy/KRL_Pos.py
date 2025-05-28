from .KRLVariable import KRLVariable


class KRLPos(KRLVariable):
    nodes = ("X", "Y", "Z", "A", "B", "C")

    def __init__(self, name):
        self.values = {}
        self.name = name
        pass

    def set_all(self, x, y, z, a, b, c):
        vals = (x, y, z, a, b, c)
        for i, node in enumerate(self.nodes):
            self.values[node] = vals[i]

    def set_x(self, x):
        self.values["X"] = x

    def set_y(self, y):
        self.values["Y"] = y

    def set_z(self, z):
        self.values["Z"] = z

    def set_a(self, a):
        self.values["A"] = a

    def set_b(self, b):
        self.values["B"] = b

    def set_c(self, c):
        self.values["C"] = c

    def __iadd__(self, other):
        for node in self.nodes:
            self.values[node] += other.values[node]
        return self

    def get_KRL_string(self):
        sb = "{"  # String builder
        for i, node in enumerate(self.nodes):
            sb += f"{node} {self.values.get(node, 0.0)}"
            # Do not print comma after last value
            if i + 1 != len(self.nodes):
                sb += ", "
        sb += "}"
        return sb

    def from_KRL_string(self, response):
        vals = response.strip("{}").split(":")[1].split(",")
        for i, node in enumerate(self.nodes):
            self.values[node] = float(vals[i].strip(node + " "))
