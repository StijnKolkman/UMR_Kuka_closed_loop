from .KRLVariable import KRLVariable

class KRLParam(KRLVariable):

    def __init__(self, name):
        self.name = name
        self.value = 0
        pass

    def set_value(self,val):
        self.value = val

    def get_KRL_string(self):
        return self.value
    
    def from_KRL_string(self,response):
        self.value = response
    