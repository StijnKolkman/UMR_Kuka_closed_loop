from ctypes import byref, c_uint, CDLL, cdll, c_int, c_long

from enum import IntEnum

# sleep function
import time
import os
import math
import sys


# Define the Motor Modes using an IntEnum
class MotorMode(IntEnum):
    ProfilePosition = 1
    ProfileVelocity = 3
    Homing = 6
    InterpolatedPosition = 7
    Position = -1
    Velocity = -2
    Current = -3
    MasterEncoder = -5
    StepDirection = -6


class Motor:
    encoder_counts_per_rotation = 2000
    gearbox_ratio = 26 / 7

    def __init__(self, NodeID, USBID):
        # Build full path to the DLL; adjust if necessary

        if sys.platform.startswith("win"):
            lib_name = "EposCmd64.dll"
        elif sys.platform.startswith("linux"):
            lib_name = "libEposCmd.so.6.8.1.0"  # or the correct name for the Linux shared library
        else:
            raise OSError("Unsupported platform")
        # Load the DLL – ensure it is available in the path specified.
        dll_path = os.path.join(os.path.dirname(__file__), "..", "include", lib_name)
        cdll.LoadLibrary(dll_path)
        self.epos = CDLL(dll_path)

        # Motor connection variables
        self.keyhandle = 0
        self.NodeID = NodeID
        self.USBID = USBID

        # Return variables
        self.ret = 0
        self.pErrorCode = c_uint()
        self.pDeviceErrorCode = c_uint()
        self.mode = None

    def WaitAcknowledged(self):
        ObjectIndex = 0x6041
        ObjectSubindex = 0x0
        NbOfBytesToRead = 0x02
        pNbOfBytesRead = c_uint()
        pData = c_uint()

        Mask_Bit12 = 0x1000
        i = 0

        while True:
            self.ret = self.epos.VCS_GetObject(
                self.keyhandle,
                self.NodeID,
                ObjectIndex,
                ObjectSubindex,
                byref(pData),
                NbOfBytesToRead,
                byref(pNbOfBytesRead),
                byref(self.pErrorCode),
            )
            # Check Bit12
            if (pData.value & Mask_Bit12) != Mask_Bit12:
                return 1  # New profile started or acknowledged
            if i > 20:
                return 0  # Timed out
            time.sleep(1)
            i += 1

    def GetPosition(self):
        ObjectIndex = 0x6064
        ObjectSubIndex = 0x00
        NbOfBytesToRead = 0x04
        NbOfBytesRead = c_uint()
        pData = c_int()

        self.ret = self.epos.VCS_GetObject(
            self.keyhandle,
            self.NodeID,
            ObjectIndex,
            ObjectSubIndex,
            byref(pData),
            NbOfBytesToRead,
            byref(NbOfBytesRead),
            byref(self.pErrorCode),
        )

        if self.ret == 1:
            print("Position Actual Value: %d [inc]" % pData.value)
            return pData.value
        else:
            print("GetObject failed")
            return None

    def GetPositionIs(self):
        # Modified to return the actual position value rather than just printing a message.
        pPositionIs = c_long()
        ret = self.epos.VCS_GetPositionIs(
            self.keyhandle, self.NodeID, byref(pPositionIs), byref(self.pErrorCode)
        )
        if ret == 1:
            # print("Position Actual Value: %d [inc]" % pPositionIs.value)
            return pPositionIs.value
        else:
            # print("GetPositionIs failed")
            return None

    def GetPositionRadians(self):
        counts_to_radians = (2 * math.pi) / (
            self.encoder_counts_per_rotation * self.gearbox_ratio
        )
        position_counts = self.GetPositionIs()
        if position_counts is not None:
            return position_counts * counts_to_radians
        else:
            return None

    def ConvertRadiansToCount(self, radians: float) -> int:
        radians_to_counts = (self.encoder_counts_per_rotation * self.gearbox_ratio) / (
            2 * math.pi
        )
        return int(round(radians * radians_to_counts))

    def OpenCommunication(self):
        # print("Opening Port...")
        self.keyhandle = self.epos.VCS_OpenDevice(
            b"EPOS4",
            b"MAXON SERIAL V2",
            b"USB",
            bytes(f"USB{self.USBID}", "utf-8"),
            byref(self.pErrorCode),
        )
        if self.keyhandle != 0:
            # print("keyhandle: %8d" % self.keyhandle)
            self.ret = self.epos.VCS_GetDeviceErrorCode(
                self.keyhandle,
                self.NodeID,
                1,
                byref(self.pDeviceErrorCode),
                byref(self.pErrorCode),
            )
            # print("Device Error: %#5.8x" % self.pDeviceErrorCode.value)
        else:
            print("Could not open Com-Port")
            print("keyhandle: %8d" % self.keyhandle)
            print("Error Opening Port: %#5.8x" % self.pErrorCode.value)

    def EnableMotor(self):
        if self.pDeviceErrorCode.value == 0:
            self.ret = self.epos.VCS_SetEnableState(
                self.keyhandle, self.NodeID, byref(self.pErrorCode)
            )
            # print("Device Enabled")
        else:
            print("epos4 is in Error State: %#5.8x" % self.pDeviceErrorCode.value)
            print(
                "epos4 Error Description can be found in the epos4 Firmware Specification"
            )

    def DisableMotor(self):
        self.ret = self.epos.VCS_SetDisableState(
            self.keyhandle, self.NodeID, byref(self.pErrorCode)
        )
        print("Device Disabled")

    def CloseCommunication(self):
        self.ret = self.epos.VCS_CloseDevice(self.keyhandle, byref(self.pErrorCode))
        print("Error Code Closing Port: %#5.8x" % self.pErrorCode.value)

    def SetOperationMode(self, mode: MotorMode):
        self.ret = self.epos.VCS_SetOperationMode(
            self.keyhandle, self.NodeID, mode.value, byref(self.pDeviceErrorCode)
        )
        if self.ret != 0:
            self.mode = mode

    def SetVelocityProfile(self, acceleration, deceleration):
        self.ret = self.epos.VCS_SetVelocityProfile(
            self.keyhandle,
            self.NodeID,
            acceleration,
            deceleration,
            byref(self.pErrorCode),
        )
        self.SetOperationMode(MotorMode.ProfileVelocity)

    def RunSetVelocity(self, velocity):
        if self.mode == MotorMode.ProfileVelocity:
            self.ret = self.epos.VCS_MoveWithVelocity(
                self.keyhandle, self.NodeID, velocity, byref(self.pErrorCode)
            )

    def SetPositionMust(self, position):
        if self.mode == MotorMode.Position:
            self.ret = self.epos.VCS_SetPositionMust(
                self.keyhandle, self.NodeID, position, byref(self.pDeviceErrorCode)
            )

    def SetPositionProfile(self, velocity, acceleration, deceleration):
        self.ret = self.epos.VCS_SetPositionProfile(
            self.keyhandle,
            self.NodeID,
            velocity,
            acceleration,
            deceleration,
            byref(self.pDeviceErrorCode),
        )
        self.SetOperationMode(MotorMode.ProfilePosition)

    def SetPosition(self, position, absolute: bool, immediately: bool):
        if self.mode == MotorMode.ProfilePosition:
            self.ret = self.epos.VCS_MoveToPosition(
                self.keyhandle,
                self.NodeID,
                position,
                absolute,
                immediately,
                byref(self.pDeviceErrorCode),
            )
