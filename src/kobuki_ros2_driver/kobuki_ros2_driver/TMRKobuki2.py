# -------------------------------------------------------------------------------------------------------------------- #
# File Name    : TMRKobuki2.py
# Project Name : ExaRobotCtrl
# Author       : Raim.Delgado
# Organization : SeoulTech
# Description  :
# [Revision History]
# >> 2021.01.24 - First Commit
# >> 2021.01.25 - Add Serial Communication
# >> 2021.01.25 - Add Send Commands except Set GPO
# >> 2021.02.08 - Add Packet Parsing
# -------------------------------------------------------------------------------------------------------------------- #
import os
import sys
import time

#원주율(회전량 계산)
from numpy import pi 
#기본 차륜로봇 클래스
from .TMRDiff import * 
#임포트 할 모듈 디렉토리 경로 추가
FILE_PATH = os.path.dirname(os.path.realpath(__file__))  # %PROJECT_ROOT%/Library/Socket
ROOT_PATH = os.path.dirname(os.path.dirname(FILE_PATH))
INCLUDE_PATH = os.path.join(ROOT_PATH, "Include")
MDI_PATH = os.path.join(INCLUDE_PATH, "MdiBackground")
DOCK_PATH = os.path.join(INCLUDE_PATH, "DockWidgets")
MISC_PATH = os.path.join(INCLUDE_PATH, "Misc")
RESOURCES_PATH = os.path.join(INCLUDE_PATH, "Resources")
LIBRARY_PATH = os.path.join(ROOT_PATH, "Library")
SERIAL_PATH = os.path.join(LIBRARY_PATH, "Serial")
sys.path.extend([FILE_PATH, ROOT_PATH, INCLUDE_PATH, MDI_PATH, DOCK_PATH, RESOURCES_PATH, LIBRARY_PATH, SERIAL_PATH])
sys.path = list(set(sys.path))
del FILE_PATH, ROOT_PATH, INCLUDE_PATH, MDI_PATH, DOCK_PATH, RESOURCES_PATH, LIBRARY_PATH, SERIAL_PATH

from .SerialPort import CSerialPort
from Commons import PySignal, is_system_win, MAX_SERIAL_BUFFER, search_pattern
from Commons import convert_u16_to_bytearray_le, convert_u32_to_bytearray_le, convert_bytearray_to_int_le
from Commons import convert_s16_to_bytearray_le, convert_s32_to_bytearray_le

#열거형 정의
@unique
class EnumKobuki2CmdIdentifier(IntEnum):
    BASE_CONTROL = 0x01
    SOUND = 0x03
    SOUND_SEQUENCE = 0x04
    REQUEST_EXTRA = 0x09
    CHANGE_FRAME = 0x0A
    REQUEST_EEPROM = 0x0B
    GPO = 0x0C
    SET_CONTROLLER_GAIN = 0x0D
    GET_CONTROLLER_GAIN = 0x0E

    @property
    def toInt(self):
        return int(self)


@unique
class EnumKobuki2FeedbackIdentifier(IntEnum):
    CORE_SENSORS = 0x01
    DOCK_INFRARED = 0x03
    INERTIAL_SENSOR = 0x04
    CLIFF = 0x05
    CURRENT = 0x06
    HW_VERSION = 0x0A
    FW_VERSION = 0x0B
    GYRO = 0x0D
    GPI = 0x10
    UDID = 0x13
    CONTROLLER_INFO = 0x15

    @property
    def toInt(self):
        return int(self)


@unique
class EnumKobuki2SoundSequence(IntEnum):
    ON = 0
    OFF = 1
    RECHARGE = 2
    BUTTON = 3
    ERROR = 4
    CLEANINGSTART = 5
    CLEANINGEND = 6

    @property
    def toInt(self):
        return int(self)


@unique
class EnumKobuki2ExtraData(IntEnum):
    HW_VERSION = 0x01
    FW_VERSION = 0x02
    UDID = 0x08

    @property
    def toInt(self):
        return int(self)

#실제 하드웨어와 통신하는 클래스(차륜형 로봇 기본모델의 속도->위치 변환 담당)
class CTMRKobuki2(CTMRDiff):
    _serial: CSerialPort
    btaSerialBuff: bytearray
    strSerialPort: str
    nSerialBaud: int
    nSerialByteSize: int
    strSerialParity: str
    fSerialStopBits: float
    fRadiusLimit: float
    fPrevTime: float
#초기화(물리파라미터 설정)
    def __init__(self, adictConfig: Union[None, dict] = None):
        if adictConfig is None:
            # this is according to the kobuki parameters
            # refer to: https://yujinrobot.github.io/kobuki/enAppendixKobukiParameters.html
            # and https://www.roscomponents.com/en/index.php?controller=attachment&id_attachment=347
            fWheelBase = 0.23
            fWheelRadius = 0.035
            nEncoderRes = 52  # PPR = 13
            fGearRatio = 49.5833
            fMaxVel = 0.7
            fMaxAcc = 0.4  # arbitrary, cannot find anywhere
            fMaxJerk = 0.2  # arbitrary, cannot find anywhere
            fMaxYaw = 1.9199  # 180 deg/s (>110 deg/s : gyro performance will degrade)
            super(CTMRKobuki2, self).__init__(fWheelRadius, fWheelBase, fMaxVel, fMaxAcc, fMaxJerk, fMaxYaw)
            self.set_wheel_spec(nEncoderRes, fGearRatio)
        else:
            # todo: add configuration file, preferably JSON
            pass
#초기화(센서 데이터 전달을 위한 신호 정의)
        # Signals #
        self.sig_core_sensors = PySignal(dict)
        self.sig_dock_ir = PySignal(int, int, int)
        self.sig_dock_inertial_sensor = PySignal(int, int)
        self.sig_cliff = PySignal(int, int, int)
        self.sig_current = PySignal(int, int)
        self.sig_get_hw_ver = PySignal(str)
        self.sig_gyro = PySignal(int, int, tuple, tuple)
        self.sig_gpi = PySignal(int, list)
        self.sig_udid = PySignal(list)
        self.sig_controller_info = PySignal(int, int, int, int)

        # Serial Communication Specifications
        self.nSerialBaud = 115200
        self.nSerialByteSize = 8
        self.strSerialParity = 'N'
        self.fSerialStopBits = 1
#초기화(시리얼통신을 위한 포트 설정 및 콜백)
        # Serial Callbacks
        self.btaSerialBuff = bytearray()
        self._serial = CSerialPort()
        self._serial.sig_connected.connect(self.on_serial_connect)
        self._serial.sig_disconnected.connect(self.on_serial_disconnect)
        self._serial.sig_serial_error.connect(self.on_serial_error)
        self._serial.sig_send_data.connect(self.on_serial_send)
        self._serial.sig_recv_data.connect(self.on_serial_recv)

        # Radius limit: Avoid 0 denominator when calculating the radius of curvature
        self.fRadiusLimit = 0.0001

        # calculate time difference to calculate velocity
        self.nPrevTimeStamp = 0
        self.fPrevRadR = 0
        self.fPrevRadL = 0

        self.IsSerial = True  # serial robot

    def release(self):
        self.disconnect()
#시리얼 통신 연결 및 해제(기본값 COM1, /dev/ttyUSB0)  
    ### Serial Communication ###
    def connect(self, strPort: Union[None, str] = None):
        if strPort is None:
            if is_system_win():
                self.strSerialPort = "COM1"
            else:
                self.strSerialPort = "/dev/ttyUSB0"
        else:
            self.strSerialPort = str(strPort)

        self._serial.setParams(port=self.strSerialPort, baudrate=self.nSerialBaud, bytesize=self.nSerialByteSize,
                               parity=self.strSerialParity, stopbits=self.fSerialStopBits)
        self._serial.connect()

    def is_connected(self) -> bool:
        return self._serial.is_connected()

    def disconnect(self):
        if self.is_connected():
            self._serial.disconnect()
#시리얼 콜백함수
    ### CALLBACKS ###
    def on_serial_connect(self):
        self.sig_connected.emit()
        self.reset_curr_pos()
        self.request_extra_data(EnumKobuki2ExtraData.FW_VERSION)

    def on_serial_disconnect(self):
        self.sig_disconnected.emit()

    def on_serial_error(self):
        self.sig_comm_error.emit()

    def on_serial_send(self, abtData: bytes):
        pass

    def on_serial_recv(self, abtData: bytes):
        if len(self.btaSerialBuff) >= MAX_SERIAL_BUFFER:
            self.btaSerialBuff.clear()

        try:
            self.btaSerialBuff.extend(bytearray(abtData))
            self.deserialize_packet()

        except Exception:
            self.btaSerialBuff.clear()
#패킷처리(정상패킷인지 판단 후 유효값만 분리해서 전달)
    def deserialize_packet(self):
        bta_stx = bytearray([0xAA, 0x55])  # STX
        n_next_packet_idx = 0

        n_idx = search_pattern(self.btaSerialBuff, bta_stx)
        if n_idx is not None:
            # length of the whole packet
            # header: 2 bytes, # length: 1 byte, minimum payload : 3 bytes, checksum: 1 byte
            if len(self.btaSerialBuff) >= n_idx + 7:
                # length of the payload
                n_payload_length = int(self.btaSerialBuff[n_idx + 2])
                if len(self.btaSerialBuff) >= n_idx + 4 + n_payload_length:
                    n_next_packet_idx = n_idx + 4 + n_payload_length
                    tmp = self.btaSerialBuff[n_idx + 3: n_next_packet_idx]  # start of payload
                    self.parse_payload_values(tmp)

        if len(self.btaSerialBuff) >= n_next_packet_idx:
            self.btaSerialBuff = self.btaSerialBuff[n_next_packet_idx:]
#센서ID로 해석 후 해당하는 센서 신호 발생
    def parse_payload_values(self, abtPayload):

        while len(abtPayload) > 0:

            if abtPayload[0] == EnumKobuki2FeedbackIdentifier.CORE_SENSORS:
                length = abtPayload[1]
                dict_core_sensors = {}
                timestamp = convert_bytearray_to_int_le(abtPayload[2:4])
                bumper = abtPayload[4]
                wheel_drop = abtPayload[5]
                cliff = abtPayload[6]
                left_encoder = convert_bytearray_to_int_le(abtPayload[7:9])
                right_encoder = convert_bytearray_to_int_le(abtPayload[9:11])
                left_pwm = abtPayload[11]
                right_pwm = abtPayload[12]
                button = abtPayload[13]
                charger = abtPayload[14]
                battery = abtPayload[15]
                overcurrent = abtPayload[16]

                dict_core_sensors['timestamp'] = timestamp
                dict_core_sensors['bumper'] = bumper
                dict_core_sensors['wheel_drop'] = wheel_drop
                dict_core_sensors['cliff'] = cliff
                dict_core_sensors['left_pwm'] = left_pwm
                dict_core_sensors['right_pwm'] = right_pwm
                dict_core_sensors['button'] = button
                dict_core_sensors['charger'] = charger
                dict_core_sensors['battery'] = battery
                dict_core_sensors['overcurrent'] = overcurrent

                # calculate position
                f_dt = (timestamp - self.nPrevTimeStamp) / 1000.
                f_rad_r = 2 * pi * (right_encoder / self.nEncRes / self.GearRatio)
                f_rad_l = 2 * pi * (left_encoder / self.nEncRes / self.GearRatio)
                f_rads_r = (f_rad_r - self.fPrevRadR) / f_dt
                f_rads_l = (f_rad_l - self.fPrevRadL) / f_dt
                stCurrPos, f_vel_c, f_omega_c = self.calculate_pose((f_rads_r, f_rads_l), f_dt)
                self.nPrevTimeStamp = timestamp
                self.fPrevRadR = f_rad_r
                self.fPrevRadL = f_rad_l

                self.sig_core_sensors.emit(dict_core_sensors)
                self.sig_get_enc.emit(right_encoder, left_encoder)
                self.sig_get_vel.emit(f_rads_r, f_rads_l)
                self.sig_get_pos.emit(stCurrPos.X_POS, stCurrPos.Y_POS, stCurrPos.THETA)
                del abtPayload[:length + 2]  # actual length + headers (ID/length)

            elif abtPayload[0] == EnumKobuki2FeedbackIdentifier.DOCK_INFRARED:
                length = abtPayload[1]
                right_signal = abtPayload[2]
                central_signal = abtPayload[3]
                left_signal = abtPayload[4]
                self.sig_dock_ir.emit(right_signal, central_signal, left_signal)
                del abtPayload[:length + 2]  # actual length + headers (ID/length)

            elif abtPayload[0] == EnumKobuki2FeedbackIdentifier.INERTIAL_SENSOR:
                length = abtPayload[1]
                angle = convert_bytearray_to_int_le(abtPayload[2:4], abIsSigned=True)
                angle_rate = convert_bytearray_to_int_le(abtPayload[4:6], abIsSigned=True)
                self.sig_dock_inertial_sensor.emit(angle, angle_rate)
                del abtPayload[:length + 2]  # actual length + headers (ID/length)

            elif abtPayload[0] == EnumKobuki2FeedbackIdentifier.CLIFF:
                length = abtPayload[1]
                right_cliff = convert_bytearray_to_int_le(abtPayload[2:4])
                central_cliff = convert_bytearray_to_int_le(abtPayload[4:6])
                left_cliff = convert_bytearray_to_int_le(abtPayload[6:8])
                self.sig_cliff.emit(right_cliff, central_cliff, left_cliff)
                del abtPayload[:length + 2]  # actual length + headers (ID/length)

            elif abtPayload[0] == EnumKobuki2FeedbackIdentifier.CURRENT:
                length = abtPayload[1]
                left_motor = abtPayload[2]
                right_motor = abtPayload[3]
                self.sig_current.emit(right_motor, left_motor)
                del abtPayload[:length + 2]  # actual length + headers (ID/length)

            elif abtPayload[0] == EnumKobuki2FeedbackIdentifier.HW_VERSION:
                length = abtPayload[1]
                patch = abtPayload[2]
                minor = abtPayload[3]
                major = abtPayload[4]
                hw_ver = str(major) + "." + str(minor) + str(patch)
                self.sig_get_hw_ver.emit(hw_ver)
                del abtPayload[:length + 2]  # actual length + headers (ID/length)
                self.request_extra_data(EnumKobuki2ExtraData.FW_VERSION)

            elif abtPayload[0] == EnumKobuki2FeedbackIdentifier.FW_VERSION:
                length = abtPayload[1]
                patch = abtPayload[2]
                minor = abtPayload[3]
                major = abtPayload[4]
                fw_ver = str(major) + "." + str(minor) + str(patch)
                self.sig_get_ver.emit(fw_ver)
                del abtPayload[:length + 2]  # actual length + headers (ID/length)
                self.request_extra_data(EnumKobuki2ExtraData.HW_VERSION)

            elif abtPayload[0] == EnumKobuki2FeedbackIdentifier.GYRO:
                length = abtPayload[1]
                frame_id = abtPayload[2]
                data_length = abtPayload[3]
                list_x, list_y, list_z = [], [], []
                list_ang_vel_x, list_ang_vel_y, list_ang_vel_z = [], [], []
                for i in range(int(data_length / 3)):
                    x_axis = convert_bytearray_to_int_le(abtPayload[4 + i * 6:6 + i * 6])
                    y_axis = convert_bytearray_to_int_le(abtPayload[6 + i * 6:8 + i * 6])
                    z_axis = convert_bytearray_to_int_le(abtPayload[8 + i * 6:10 + i * 6])
                    list_x.append(x_axis)
                    list_y.append(y_axis)
                    list_z.append(z_axis)
                    #  deg/s ratio is 0.00875
                    #  rotated 90 degree counterclockwise about z-axis
                    f_ang_vel_x = 0.00875 * y_axis * -1
                    f_ang_vel_y = 0.00875 * x_axis
                    f_ang_vel_z = 0.00875 * z_axis
                    list_ang_vel_x.append(f_ang_vel_x)
                    list_ang_vel_y.append(f_ang_vel_y)
                    list_ang_vel_z.append(f_ang_vel_z)

                tuple_raw = (list_x, list_y, list_z)
                tuple_ang_vel = (list_ang_vel_x, list_ang_vel_y, list_ang_vel_z)

                self.sig_gyro.emit(frame_id, int(data_length/3), tuple_raw, tuple_ang_vel)
                del abtPayload[:length + 2]  # actual length + headers (ID/length)

            elif abtPayload[0] == EnumKobuki2FeedbackIdentifier.GPI:
                length = int(abtPayload[1])
                digital_input = convert_bytearray_to_int_le(abtPayload[2:4])
                list_analog_input = []
                for i in range(4):
                    list_analog_input.append(convert_bytearray_to_int_le(abtPayload[4 + i * 2:6 + i * 2]))

                self.sig_gpi.emit(digital_input, list_analog_input)
                del abtPayload[:length + 2]  # actual length + headers (ID/length)

            elif abtPayload[0] == EnumKobuki2FeedbackIdentifier.UDID:
                length = int(abtPayload[1])
                list_udid = []
                for i in range(3):
                    list_udid.append(convert_bytearray_to_int_le(abtPayload[2 + i * 4:6 + i * 4]))

                self.sig_udid.emit(list_udid)
                del abtPayload[:length + 2]  # actual length + headers (ID/length)

            elif abtPayload[0] == EnumKobuki2FeedbackIdentifier.CONTROLLER_INFO:
                length = int(abtPayload[1])
                type = abtPayload[2]
                p_gain = convert_bytearray_to_int_le(abtPayload[3:7])
                i_gain = convert_bytearray_to_int_le(abtPayload[7:11])
                d_gain = convert_bytearray_to_int_le(abtPayload[11:15])

                self.sig_controller_info.emit(type, p_gain, i_gain, d_gain)
                del abtPayload[:length + 2]  # actual length + headers (ID/length)

            else:
                abtPayload.pop()

    ### COMMANDS ###
#twist타입 cmd_vel을 속도, 회전반지름으로 변환     
    def set_velocity_control(self, afVel: float, afYawRate: float):
        bta_payload = bytearray([])

        # avoid zero denominator (undefined radius)
        # translation motion only
        if abs(afYawRate) < self.fRadiusLimit:
            f_radius = 0.
            f_vel = afVel * 1000.
        else:
            f_radius = afVel * 1000. / afYawRate

            # rotation only
            if abs(afVel) < self.fRadiusLimit or abs(f_radius) <= 1.:
                f_vel = 1000. * self.WheelBase * afYawRate / 2.
                f_radius = 1.
            else:
                if f_radius > 0.:
                    f_vel = (f_radius + 1000. * self.WheelBase / 2.) * afYawRate
                else:
                    f_vel = (f_radius - 1000. * self.WheelBase / 2.) * afYawRate

        bta_payload.extend(convert_s16_to_bytearray_le(int(f_vel)))
        bta_payload.extend(convert_s16_to_bytearray_le(int(f_radius)))
        self.send_packet(EnumKobuki2CmdIdentifier.BASE_CONTROL, 4, bta_payload)
#중심속도, 회전율 기반으로 이동
    def move_joint_space(self, afRight: float, afLeft: float):
        vel_c, omega_c = self.calculate_center_vel((afRight, afLeft))
        self.set_velocity_control(vel_c, omega_c)
#정지
    def move_stop(self):
        self.set_velocity_control(0., 0.)
#사운드 재생
    def play_custom_sound(self, anFreqHz: int, anDurationMs: int):
        bta_payload = bytearray([])
        f_alpha = 0.00000275

        if anDurationMs < 0:
            anDurationMs = 0

        if anDurationMs > 255:
            anDurationMs = 255

        n_note = int(1 / (anFreqHz * f_alpha))
        bta_payload.extend(convert_u16_to_bytearray_le(n_note))
        bta_payload.append(anDurationMs)
        self.send_packet(EnumKobuki2CmdIdentifier.SOUND, 3, bta_payload)

    def play_sound_sequence(self, anSequenceNo: int):
        bta_payload = bytearray([])

        if anSequenceNo < 0:
            anSequenceNo = 0

        if anSequenceNo > 6:
            anSequenceNo = 6

        bta_payload.append(anSequenceNo)
        self.send_packet(EnumKobuki2CmdIdentifier.SOUND_SEQUENCE, 1, bta_payload)

    def request_extra_data(self, aenumExtra: EnumKobuki2ExtraData = EnumKobuki2ExtraData.UDID):
        bta_payload = bytearray([])
        try:
            en_data = EnumKobuki2ExtraData(aenumExtra)
        except ValueError:
            en_data = EnumKobuki2ExtraData.UDID

        bta_payload.extend(convert_u16_to_bytearray_le(int(en_data)))
        self.send_packet(EnumKobuki2CmdIdentifier.REQUEST_EXTRA, 2, bta_payload)

    # todo: add this feature
    def set_gpio(self):
        pass
#속도제어 품질 조절
    def set_controller_gain(self, afKp: float = 100, afKi: float = 0.1, afKd: float = 2):
        bta_payload = bytearray([])
        n_type = 1
        if afKp == 100 and afKi == 0.1 and afKd == 2:
            n_type = 0

        n_Kp = int(afKp * 1000.)
        n_Ki = int(afKi * 1000.)
        n_Kd = int(afKd * 1000.)

        bta_payload.append(n_type)
        bta_payload.extend(convert_u32_to_bytearray_le(n_Kp))
        bta_payload.extend(convert_u32_to_bytearray_le(n_Ki))
        bta_payload.extend(convert_u32_to_bytearray_le(n_Kd))

        self.send_packet(EnumKobuki2CmdIdentifier.SET_CONTROLLER_GAIN, 13, bta_payload)

    def get_controller_gain(self):
        bta_payload = bytearray([])
        bta_payload.append(0x01)  # dummy
        self.send_packet(EnumKobuki2CmdIdentifier.GET_CONTROLLER_GAIN, 1, bta_payload)

    # Serialize
    def make_payload(self, anCmd: int, anSize: int, abtaPayLoad: bytearray = None) -> bytearray:
        buff = bytearray([])
        cmd = int(anCmd)
        size = int(anSize)
        buff.append(cmd)
        buff.append(size)

        if abtaPayLoad is not None:
            buff.extend(abtaPayLoad)

        return buff

    def make_packet(self, abtaPayload: bytearray) -> bytearray:
        buff = bytearray([0xAA, 0x55])  # STX
        buff.append(len(abtaPayload))
        buff.extend(abtaPayload)
        tmp = buff[2:]  # remove STX for checksum calculation
        buff.append(self.calculate_checksum(tmp))

        return buff

    def send_packet(self, anCmd: int, anSize: int, abtaPayLoad: bytearray = None) -> bool:
        bta_payload = self.make_payload(anCmd, anSize, abtaPayLoad)
        bta_packet = self.make_packet(bta_payload)

        if self._serial.is_connected():
            self._serial.sendData(bta_packet)
            return True
        else:
            return False

    def calculate_checksum(self, abtaPayload: bytearray) -> Union[int, bool]:
        if len(abtaPayload) == 0:
            return False

        tmp = 0
        for data in abtaPayload:
            tmp ^= data

        return tmp


if __name__ == '__main__':
    test = CTMRKobuki2()
    test.connect()
    test.get_curr_pos_t()
    test.set_velocity_control(-0.1, 0)
