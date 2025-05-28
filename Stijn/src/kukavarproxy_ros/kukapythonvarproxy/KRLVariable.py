class KRLVariable:
    def send(self, kuka):
        kuka.write(self.name, self.get_KRL_string())

    def read(self, kuka):
        res = kuka.read(self.name)
        self.from_KRL_string(res)
