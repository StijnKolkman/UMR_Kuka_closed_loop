from .KRL_Parameter import KRLParam
from .KRL_Pos import KRLPos

_tool = KRLPos("COM_FRAME")
_action = KRLParam("COM_ACTION")
_rounding = KRLParam("COM_ROUNDM")
_velCP = KRLParam("COM_VALUE1")
_velORI = KRLParam("COM_VALUE2")
_accCP = KRLParam("COM_VALUE3")
_accORI = KRLParam("COM_VALUE4")


def read_action(kuka):
    _action.read(kuka)
    return _action.value

def set_tool(x, y, z, a, b, c):
    _tool.set_all(x, y, z, a, b, c)

def send_tool(kuka):
    _tool.send(kuka)
    _action.set_value(5)
    _action.send(kuka)

def send_joint_move(kuka):
    _action.set_value(15)
    _action.send(kuka)

def set_rounding(val):
    _rounding.set_value(val)

def send_rounding(kuka):
    _rounding.send(kuka)
    _action.set_value(8)
    _action.send(kuka)

def set_velocity(velCP, velORI, accCP, accORI):
    _velCP.set_value(velCP)
    _velORI.set_value(velORI)
    _accCP.set_value(accCP)
    _accORI.set_value(accORI)

def set_velCP(velCP):
    _velCP.set_value(velCP)

def set_velORI(velORI):
    _velORI.set_value(velORI)

def set_accCP(accCP):
    _accCP.set_value(accCP)

def set_accORI(accORI):
    _accORI.set_value(accORI)

def send_velocity(kuka):
    _velCP.send(kuka)
    _velORI.send(kuka)
    _accCP.send(kuka)
    _accORI.send(kuka)
    _action.set_value(7)
    _action.send(kuka)
