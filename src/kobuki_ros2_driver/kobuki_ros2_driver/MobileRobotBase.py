import os
import sys
from enum import IntEnum, auto, unique
from typing import Tuple, Union
from dataclasses import dataclass  # for python < 3.7, dataclasses should be installed : python -m pip install dataclasses

FILE_PATH = os.path.dirname(os.path.realpath(__file__))  # %PROJECT_ROOT%/Library/Socket
ROOT_PATH = os.path.dirname(os.path.dirname(FILE_PATH))
INCLUDE_PATH = os.path.join(ROOT_PATH, "Include")
MDI_PATH = os.path.join(ROOT_PATH, "MdiBackground")
DOCK_PATH = os.path.join(ROOT_PATH, "DockWidgets")
MISC_PATH = os.path.join(ROOT_PATH, "Misc")
RESOURCES_PATH = os.path.join(ROOT_PATH, "Resources")
sys.path.extend([FILE_PATH, ROOT_PATH, INCLUDE_PATH, MDI_PATH, DOCK_PATH, RESOURCES_PATH])
sys.path = list(set(sys.path))
del FILE_PATH, ROOT_PATH, INCLUDE_PATH, MDI_PATH, DOCK_PATH, RESOURCES_PATH

from .Commons import PySignal


@unique
class EnumDriveType(IntEnum):
    NOT_A_ROBOT = 0
    DIFFERENTIAL = auto()
    CAR_TYPE = auto()
    OMNIDIRECTIONAL = auto()
    SYNCRHO = auto()
    MECANUM = auto()

    @property
    def toInt(self):
        return int(self)


@unique
class EnumJogDirection(IntEnum):
    JOG_FW = 0
    JOG_BW = auto()
    JOG_LEFT = auto()
    JOG_RIGHT = auto()
    JOG_FW_LEFT = auto()
    JOG_FW_RIGHT = auto()
    JOG_BW_LEFT = auto()
    JOG_BW_RIGHT = auto()

    @property
    def toInt(self):
        return int(self)


@unique
class EnumFKMethod(IntEnum):
    EULER = 0
    RUNGE_KUTTA = auto()
    EXACT = auto()

    @property
    def toInt(self):
        return int(self)


@dataclass  # for python < 3.7, dataclasses should be installed : python -m pip install dataclasses
class RobotPose:
    X_POS: float = 0.
    Y_POS: float = 0.
    Z_POS: float = 0.  # not used in 2D Navigation
    THETA: float = 0.  # heading angle in radians


@dataclass  # for python < 3.7, dataclasses should be installed : python -m pip install dataclasses
class PhyLimits:
    MAX_VEL: float = 0.     # maximum velocity in m/s
    MAX_ACC: float = 0.     # maximum acceleration in m/s^2
    MAX_JERK: float = 0.    # maximum jerk in m/s^3
    MAX_YAW: float = 0.     # maximum yaw rate in rad/s


class CMobileRobot(object):
    """ CMobileRobot은 모바일 로봇에 대한 클래스이다.

        Args:
            anDriveType (Union[int, EnumDriveType]): 모바일 로봇의 구동타입
            anWheelNo (int): 바퀴 갯수
            afWheelRad (float): 바퀴 반지름
            afWheelBase (float): 바퀴 사이 거리
    """
    # Serial Robot
    bIsSerial = False

    # Basics
    nWheelNo: int
    """
    휠 개수
    """
    fWheelRad: float  # in meters
    """
    휠 반지름
    """
    fWheelBase: float  # in meters
    """
    휠간 거리
    """

    # Robot dimensions
    fLength: float
    """
    로봇 길이
    """
    fWidth: float
    """
    로봇 너비
    """

    # Payload and actual weight (in kg)
    fWeight: float
    """
    로봇 무계
    """
    fPayLoad: float
    """
    로봇 Payload
    """

    # drive type and physical limits
    enumDriveType: EnumDriveType  # Default is DIFFERENTIAL
    """
    구동계 타입
    """
    stPhyLimits: PhyLimits  # Default are zero-values
    """
    로봇의 PhyLimits
    가속도, 선속도, 각속도, 저크
    """
   

    # Position
    stCurrPos: RobotPose
    

    # Actual robot specs
    nEncRes: int
    """
    모터 엔코더 레졸루션
    """
    fGearRatio: float
    """
    모터 기어비
    """

    def __init__(self, anDriveType: Union[int, EnumDriveType], anWheelNo: int, afWheelRad: float, afWheelBase: float):

        """ Constructor of the Mobile Robot base class.
        User should specify the drive type, no of wheels, radius of wheel, and the distance between the wheels
        """

        super(CMobileRobot, self).__init__()
        # Basics
        self.WheelNo = anWheelNo
        self.WheelRad = afWheelRad
        self.WheelBase = afWheelBase

        ''' Initiate Variables '''
        # Robot dimensions
        self.Length = 0.
        self.Width = 0.

        # Payload and actual weight (in kg)
        self.Weight = 0.
        self.PayLoad = 0.

        # Physical Limits
        self.PhyLimits = (0., 0., 0., 0.)
        self.DriveType = anDriveType

        # Position Related
        self.stCurrPos = RobotPose(0., 0., 0., 0.)

        # Actual robot specs
        self.EncoderRes = 1
        self.GearRatio = 1

        ''' Common Signals '''
        self.sig_connected = PySignal()
        self.sig_disconnected = PySignal()
        self.sig_comm_error = PySignal()
        self.sig_get_enc = PySignal(int, int)
        self.sig_get_vel = PySignal(float, float)
        self.sig_get_wheel_pos = PySignal(float, float)
        self.sig_get_pos = PySignal(float, float, float)
        self.sig_get_state = PySignal(int)
        self.sig_get_ver = PySignal(str)

    def set_serial_robot(self, abIsSerial: bool):
        self.bIsSerial = abIsSerial

    def is_serial_robot(self) -> bool:
        return self.bIsSerial

    def reset_curr_pos(self):
        self.stCurrPos = RobotPose(0., 0., 0., 0.)

    def get_curr_pos(self) -> RobotPose:
        """ Returns the current position in the cartesian plane using the class XYPos
        """
        return self.stCurrPos

    def get_curr_pos_t(self) -> Tuple[float, float, float, float]:
        """ Returns a tuple of the current position in the cartesian plane (x, y, z, theta)
        """
        return self.stCurrPos.X_POS, self.stCurrPos.Y_POS, self.stCurrPos.Z_POS, self.stCurrPos.THETA

    def calculate_joint_space_vel(self, afVelCenter: float, afAngVelCenter: float) -> tuple:
        """ calculates the joint space velocities of the mobile robot.
        Returns a tuple of the calculated joint space velocities
        """
        pass

    def calculate_center_vel(self, atupleAngVels: tuple) -> tuple:
        """ calculates the center velocities of the mobile robot.
        Returns a tuple of central linear vel and yaw rate
        """
        pass

    def calculate_pose_actual(self, atupleAngVels: tuple, afDeltaT: float,
                              aenumFKMethod: EnumFKMethod = EnumFKMethod.EULER) -> RobotPose:
        """ calculates the forward kinematics of the mobile robot considering gear ration and encoder resolution.
        fJointsVel should be a tuple of the joint space velocities
        Returns the calculated position in the cartesian plane using the class XYPos.
        This should be called everytime the robot changes position.
        """
        return self.stCurrPos

    def calculate_pose(self, atupleAngVels: tuple, afDeltaT: float,
                       aenumFKMethod: EnumFKMethod = EnumFKMethod.EULER) -> RobotPose:
        """ calculates the forward kinematics of the mobile robot.
        fJointsVel should be a tuple of the joint space velocities
        Returns the calculated position in the cartesian plane using the class XYPos.
        This should be called everytime the robot changes position.
        """
        return self.stCurrPos

    def move_jog(self, anDirection: EnumJogDirection, nJogDistance: float = 500) -> bool:
        """ moves the mobile robot on the speicifed direction (anDirection) with the configured
        jog distance (nJogDistance) in millimeters (mm). The default jog is 500 mm.
        """
        return False

    def set_phy_limits(self, afMaxVel: float, afMaxAcc: float, afMaxJerk: float, afMaxYaw: float):
        """ sets the physical limits of the robots (max velocity, max acceleration, max jerk, and max yaw)
        """
        self.stPhyLimits = PhyLimits(afMaxVel, afMaxAcc, afMaxJerk, afMaxYaw)

    def get_phy_limits(self) -> PhyLimits:
        """ returns the physical limits of the robots (max velocity, max acceleration, max jerk, and max yaw)
        using the PhyLimit dataclass
        """
        return self.stPhyLimits

    def get_phy_limits_t(self) -> Tuple[float, float, float, float]:
        """ returns a tuple of the physical limits of the robots (max velocity, max acceleration, and max jerk)
        """
        return self.stPhyLimits.MAX_VEL, self.stPhyLimits.MAX_ACC, self.stPhyLimits.MAX_JERK, self.stPhyLimits.MAX_YAW

    def set_wheel_spec(self, anEncoderRes: int, afGearRatio: float):
        """ sets the wheel spec of the robot (Encoder resolution, Gear ratio)
        """
        self.EncoderRes = anEncoderRes
        self.GearRatio = afGearRatio

    def get_wheel_spec(self) -> Tuple[int, float]:
        """ returns a tuple of the wheel specs (Encoder resolution, Gear ratio)
        """
        return self.EncoderRes, self.GearRatio

    ''' Properties '''

    @property
    def WheelNo(self) -> int:
        return self.nWheelNo

    @WheelNo.setter
    def WheelNo(self, anWheelNo: int):
        self.nWheelNo = anWheelNo

    @property
    def WheelRad(self) -> float:
        return self.fWheelRad

    @WheelRad.setter
    def WheelRad(self, afWheelRad: float):
        self.fWheelRad = afWheelRad

    @property
    def WheelBase(self) -> float:
        return self.fWheelBase

    @WheelBase.setter
    def WheelBase(self, afWheelBase: float):
        self.fWheelBase = afWheelBase

    @property
    def Length(self) -> float:
        return self.fLength

    @Length.setter
    def Length(self, afLength: float):
        self.fLength = afLength

    @property
    def Width(self) -> float:
        return self.fWidth

    @Width.setter
    def Width(self, afWidth: float):
        self.fWidth = afWidth

    @property
    def Weight(self) -> float:
        return self.fWeight

    @Weight.setter
    def Weight(self, afWeight: float):
        self.fWeight = afWeight

    @property
    def PayLoad(self) -> float:
        return self.fPayLoad

    @PayLoad.setter
    def PayLoad(self, afPayLoad: float):
        self.fPayLoad = afPayLoad

    @property
    def PhyLimits(self) -> PhyLimits:
        return self.get_phy_limits()

    @PhyLimits.setter
    def PhyLimits(self, afMaxVelAccJerk: Tuple[float, float, float, float]):
        self.set_phy_limits(afMaxVelAccJerk[0], afMaxVelAccJerk[1], afMaxVelAccJerk[2], afMaxVelAccJerk[3])

    @property
    def PhyLimitsT(self) -> Tuple[float, float, float, float]:
        return self.get_phy_limits_t()

    @property
    def DriveType(self) -> EnumDriveType:
        """ returns the drive type of the mobile robot
        """
        return self.enumDriveType

    @DriveType.setter
    def DriveType(self, anDriveType: Union[int, EnumDriveType]):
        # Ensure that the robot type is specified with EnumDriveType class. Default value is DIFFERENTIAL.
        try:
            if not isinstance(anDriveType, EnumDriveType):
                self.enumDriveType = EnumDriveType(anDriveType)
        except ValueError:
            self.enumDriveType = EnumDriveType.DIFFERENTIAL

    @property
    def EncoderRes(self) -> int:
        return self.nEncRes

    @EncoderRes.setter
    def EncoderRes(self, anEncoderRes: int):
        self.nEncRes = anEncoderRes

    @property
    def GearRatio(self) -> float:
        return self.fGearRatio

    @GearRatio.setter
    def GearRatio(self, afGearRatio: float):
        self.fGearRatio = afGearRatio

    @property
    def IsSerial(self) -> bool:
        return self.bIsSerial

    @IsSerial.setter
    def IsSerial(self, abIsSerial: bool):
        self.bIsSerial = abIsSerial
