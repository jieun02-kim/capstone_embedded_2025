import datetime
import os
import sys
import psutil
from typing import Union
import logging
from logging import handlers
import threading
import numpy as np
import ipaddress
import socket
import platform
import traceback
import ctypes
from ctypes import wintypes
import cv2
import importlib.util

FILE_PATH = os.path.dirname(os.path.realpath(__file__))  # %PROJECT_ROOT%/Include
ROOT_PATH = os.path.dirname(FILE_PATH)
INCLUDE_PATH = os.path.join(ROOT_PATH, "Include")
RESOURCES_PATH = os.path.join(ROOT_PATH, "Resources")
sys.path.extend(
    [INCLUDE_PATH, RESOURCES_PATH])
sys.path = list(set(sys.path))
del FILE_PATH, ROOT_PATH, INCLUDE_PATH, RESOURCES_PATH

MAX_SERIAL_BUFFER = 1024  # 1kb
MAX_SIZE = 100 * 1024 * 1024
glob_logger = None

DICT_COMM_TYPE = {
    0: "RS-232",
    1: "RS-485",
    2: "Ethernet",
    3: "EtherCAT",
}

DICT_AXIS_TYPE = {
    0: "Revolute",
    1: "Continuous",
    2: "Prismatic",
    3: "Planar",
    4: "Fixed",
    5: "Floating",
    6: "Jointless",
    7: "Sensor",
}


def set_cpu_affinity(pid = 0,affinity_mask={1, 2}):
    os.sched_setaffinity(pid,affinity_mask)


def check_argument_type(obj, arg):
    if type(obj) == arg:
        return True
    if obj is None:  # TODO:
        return True
    if arg == object:
        return True
    if arg in obj.__class__.__bases__:
        return True
    return False


def is_ros_installed() -> bool:
    ros_reqs = 'rclpy'
    spec = importlib.util.find_spec(ros_reqs)
    if spec is None:
        return False
    else:
        return True


def is_system_win() -> bool:
    import platform
    if platform.system() == 'Windows':
        return True

    return False


def get_ip_addr_list():
    result = []
    netlst = psutil.net_if_addrs()
    for key in netlst.keys():
        if 'loopback' not in key.lower():
            adap = key
            mac, ipv4, ipv6 = '', '', ''
            sniclist = netlst[key]
            for snic in sniclist:
                family = int(snic[0])
                address = snic[1]
                if family == -1 or family == 18:
                    mac = address
                elif family == 2:
                    ipv4 = address
                elif family == 23 or family == 30:
                    ipv6 = address
            result.append({'name': adap, 'mac': mac, 'ipv4': ipv4, 'ipv6': ipv6})
    return result


def convert_ipaddr_to_int(ipaddr: str) -> int:
    return int(ipaddress.ip_address(ipaddr))


def convert_int_to_ipaddr(ipaddr: int) -> str:
    return str(ipaddress.ip_address(ipaddr))


def convert_int_to_mac(macaddr: int) -> str:
    hex_str = '{:012x}'.format(macaddr)
    mac = ":".join(hex_str[i:i + 2] for i in range(0, 12, 2))
    return mac.upper()


def convert_mac_to_int(macddr: str) -> int:
    return int(macddr.replace(":", ""), 16)


def int_to_mac(int_val):
    hex_str = '{:012x}'.format(int_val)
    mac = ":".join(hex_str[i:i + 2] for i in range(0, 12, 2))
    return mac


class PySignal(object):
    _args = None
    _callback = None

    def __init__(self, *args):
        self._args = args

    def connect(self, callback):
        self._callback = callback

    def emit(self, *args):
        if len(args) != len(self._args):
            raise Exception('Callback::Argument Length Mismatch')
        arglen = len(args)
        if arglen > 0:
            validTypes = [check_argument_type(args[i], self._args[i]) for i in range(arglen)]
            if sum(validTypes) != arglen:
                raise Exception('Callback::Argument Type Mismatch (Definition: {}, Call: {}, Result: {})'.format(
                    self._args, args, validTypes))
        if self._callback is not None:
            self._callback(*args)


def convert_u32_to_bytearray_be(anVal: int) -> bytearray:
    if anVal > 4294967295:
        anVal = 4294967295

    if anVal < 0:
        anVal = 0

    buff = bytearray([])
    buff.append((anVal >> 24) & 0xFF)
    buff.append((anVal >> 16) & 0xFF)
    buff.append((anVal >> 8) & 0xFF)
    buff.append(anVal & 0xFF)
    return buff


def convert_s32_to_bytearray_be(anVal: int) -> bytearray:
    if anVal > 2147483647:
        anVal = 2147483647

    if anVal < -2147483648:
        anVal = -2147483648

    if anVal < 0:
        anVal = ~(anVal * -1) + 1  # 2's complement

    buff = bytearray([])
    buff.append((anVal >> 24) & 0xFF)
    buff.append((anVal >> 16) & 0xFF)
    buff.append((anVal >> 8) & 0xFF)
    buff.append(anVal & 0xFF)
    return buff


def convert_u16_to_bytearray_be(anVal: int) -> bytearray:
    if anVal > 65535:
        anVal = 65535

    if anVal < 0:
        anVal = 0

    buff = bytearray([])
    buff.append((anVal >> 8) & 0xFF)
    buff.append(anVal & 0xFF)
    return buff


def convert_s16_to_bytearray_be(anVal: int) -> bytearray:
    if anVal > 32767:
        anVal = 32767

    if anVal < -32768:
        anVal = -32768

    if anVal < 0:
        anVal = ~(anVal * -1) + 1  # 2's complement

    buff = bytearray([])
    buff.append((anVal >> 8) & 0xFF)
    buff.append(anVal & 0xFF)
    return buff


def convert_u32_to_bytearray_le(anVal: int) -> bytearray:
    if anVal > 4294967295:
        anVal = 4294967295

    if anVal < 0:
        anVal = 0

    buff = bytearray([])
    buff.append(anVal & 0xFF)
    buff.append((anVal >> 8) & 0xFF)
    buff.append((anVal >> 16) & 0xFF)
    buff.append((anVal >> 24) & 0xFF)
    return buff


def convert_s32_to_bytearray_le(anVal: int) -> bytearray:
    if anVal > 2147483647:
        anVal = 2147483647

    if anVal < -2147483648:
        anVal = -2147483648

    if anVal < 0:
        anVal = ~(anVal * -1) + 1  # 2's complement

    buff = bytearray([])
    buff.append(anVal & 0xFF)
    buff.append((anVal >> 8) & 0xFF)
    buff.append((anVal >> 16) & 0xFF)
    buff.append((anVal >> 24) & 0xFF)
    return buff


def convert_u16_to_bytearray_le(anVal: int) -> bytearray:
    if anVal > 65535:
        anVal = 65535

    if anVal < 0:
        anVal = 0

    buff = bytearray([])
    buff.append(anVal & 0xFF)
    buff.append((anVal >> 8) & 0xFF)
    return buff


def convert_s16_to_bytearray_le(anVal: int) -> bytearray:
    if anVal > 32767:
        anVal = 32767

    if anVal < -32768:
        anVal = -32768

    if anVal < 0:
        anVal = ~(anVal * -1) + 1  # 2's complement

    buff = bytearray([])
    buff.append(anVal & 0xFF)
    buff.append((anVal >> 8) & 0xFF)
    return buff


def convert_bytearray_to_int_be(abtaData: bytearray, abIsSigned=False) -> int:
    if isinstance(abtaData, int):
        return int(abtaData)

    num = 0
    for b in abtaData:
        num <<= 8
        num |= int(b) & 0XFF

    if abIsSigned:
        int_max = 2 ** (8 * len(abtaData))
        upper_limit = int((int_max / 2) - 1)

        if num > upper_limit:
            num = -(int_max - num)

    return num


def convert_bytearray_to_int_le(abtaData: bytearray, abIsSigned=False) -> int:
    if isinstance(abtaData, int):
        return int(abtaData)

    abtaData.reverse()
    return convert_bytearray_to_int_be(abtaData, abIsSigned)


def search_pattern(src, pattern) -> Union[None, int]:
    nRange = len(src) - len(pattern) + 1
    for i in range(nRange):
        if src[i] != pattern[0]:
            continue

        for j in range(len(pattern) - 1, 0, -1):
            if (src[i + j]) != pattern[j]:
                break

            if j == 1:
                return i

    return None


def convert_str_to_bytearray(strData: str) -> bytearray:
    sData = bytearray([0x00])
    if isinstance(strData, str):  # String to Bytes: Ascii
        tmp = bytearray()
        tmp.extend(map(ord, strData))
        sData = tmp

    return sData


def convert_bytearray_to_str(abtaData: bytearray) -> str:
    if isinstance(abtaData, str):
        return abtaData
    else:
        return str(abtaData.decode())


def timestamp_to_string(timestamp: datetime.datetime):
    h = timestamp.hour
    m = timestamp.minute
    s = timestamp.second
    # ms = timestamp.microsecond // 1000
    us = timestamp.microsecond
    return '%02d:%02d:%02d.%06d' % (h, m, s, us)


def get_curr_datetime():
    return datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S')


def get_curr_time():
    return '<%s>' % timestamp_to_string(datetime.datetime.now())


def write_log(strMsg: str, obj: object = None, logfile: bool = True):
    global glob_logger
    if glob_logger is None:
        # curpath = os.path.dirname(os.path.abspath(__file__))
        # dirpath = os.path.dirname(curpath)
        if getattr(sys, 'frozen', False):
            curpath = os.path.dirname(sys.executable)
        elif __file__:
            curpath = os.path.dirname(__file__)

        dirpath = os.path.dirname(curpath)
        logpath = os.path.join(dirpath, 'Log')
        if not os.path.isdir(logpath):
            os.mkdir(logpath)
        logfilepath = os.path.join(logpath, 'Console.log')
        glob_logger = logging.getLogger('console')
        fh = logging.handlers.RotatingFileHandler(logfilepath, maxBytes=MAX_SIZE, backupCount=10, encoding='utf-8')
        formatter = logging.Formatter('[%(asctime)s]%(message)s')
        fh.setFormatter(formatter)
        glob_logger.addHandler(fh)
        glob_logger.setLevel(logging.DEBUG)

    strTime = get_curr_time()
    if obj is not None:
        if isinstance(obj, threading.Thread):
            if obj.ident is not None:
                strObj = ' [%s][Thread ID:0x%x]' % (type(obj).__name__, obj.ident)
            else:
                strObj = ' [%s]' % type(obj).__name__
        else:
            strObj = ' [%s]' % type(obj).__name__
    else:
        strObj = ''
    print(strTime + strObj + ' ' + strMsg)
    if logfile:
        glob_logger.info(strObj + ' ' + strMsg)


def ecat_state2str(anState: int) -> str:
    if anState == 1:
        ecat_state = "INIT"
    elif anState == 2:
        ecat_state = "PREOP"
    elif anState == 4:
        ecat_state = "SAFEOP"
    elif anState == 8:
        ecat_state = "OP"
    else:
        ecat_state = "NONE"

    return ecat_state


def ecat_domainstate2str(anDomainState: int) -> str:
    domain_state = "NO EXCHANGED"
    if anDomainState == 1:
        domain_state = "INCOMPLETE"
    elif anDomainState == 2:
        domain_state = "COMPLETE"
    else:
        domain_state = "NO EXCHANGED"

    return domain_state


def axis_type2str(anAxisType: int) -> str:
    try:
        return DICT_AXIS_TYPE[anAxisType]
    except:
        return "UNKNOWN AXIS"


def comm_type2str(anCommType: int) -> str:
    try:
        return DICT_COMM_TYPE[anCommType]
    except:
        return "UNKNOWN COMM"


def float_to_str(f_val: float, precision: int = 4):
    str_prec = "{:.%sf}" % precision
    str_val = str_prec.format(f_val)

    return str_val


def get_closest_multiple(num: int, multiplier: int):
    if num < multiplier:
        return multiplier

    num += int(multiplier / 2)
    num -= (num % multiplier)

    return num


class CRingBuffer(object):
    def __init__(self, size_max, default_value=0.0, dtype=float, overflow=None):
        """initialization"""
        self.clear(size_max, default_value, dtype, overflow)

    def clear(self, size_max=None, default_value=None, dtype=None, overflow=None):
        """clear ring"""
        if size_max is not None:
            self.size_max = size_max

        if default_value is not None:
            self.default_value = default_value

        if dtype is not None:
            self.dtype = dtype

        if overflow is not None:
            self.overflow = overflow

        if not isinstance(default_value, np.ndarray):
            self._data = np.empty(size_max, dtype=dtype)
            self._data.fill(default_value)
        else:
            if len(default_value) == size_max:
                self._data = default_value
            else:
                msg = "len(default_value)=%d but size_max=%d, but they should be equal" % (len(default_value), size_max)
                raise NotImplementedError(msg)

        self.size = 0

        self.full = False
        self.append = self._append_not_full

    def _append_not_full(self, value):
        """append an element"""
        self._data = np.roll(self._data, 1)
        self._data[0] = value

        self.size += 1

        if self.size == self.size_max:
            self.full = True
            self.append = self._append_full
            self.overflow = self.overflow(self)

    def _append_full(self, value):
        """append an element when buffer is full"""
        self._data = np.roll(self._data, 1)
        self._data[0] = value

    @property
    def all(self):
        """return a list of elements from the oldest to the newest (len: size_max)"""
        return self._data

    @property
    def partial(self):
        """return a list of elements from the oldest to the newest (len: size)"""
        return self.all[0:self.size]

    def view(self, *args, **kwargs):
        return self.partial[::-1].view(*args, **kwargs)

    def __len__(self):
        """return size (not size_max)"""
        return self.size

    def __getitem__(self, key):
        """get element"""
        return self._data[key]

    def __repr__(self):
        """return string representation"""
        s = """<%s
    all:     %s
    partial: %s
    size/size_max: %d / %d
>""" % (self.__class__.__name__,
        self.all.__repr__(),
        self.partial.__repr__(),
        self.size, self.size_max)
        return s

    def overflow(self, *args, **kwargs):
        return

    def min(self, all=False):
        """return min"""
        if all:
            return self.all.min()
        else:
            return self.partial.min()

    def max(self, all=True):
        """return max"""
        if all:
            return self.all.max()
        else:
            return self.partial.max()


def ensurePathExist(path: str):
    targetpath = os.path.abspath(path)
    if not os.path.isdir(targetpath):
        if os.name == 'nt':
            pathsplit = targetpath.split('\\')
            ptemp = str(pathsplit[0]) + '\\'
        else:
            pathsplit = targetpath.split('/')
            ptemp = str(pathsplit[0]) + '/'
        for i in range(len(pathsplit) - 1):
            p = pathsplit[i + 1]
            ptemp = os.path.join(ptemp, p)
            ptemp = os.path.abspath(ptemp)
            if not os.path.isdir(ptemp):
                os.mkdir(ptemp)


class ErrorHandler:
    _logger = None
    _logFilePath = ""

    def __init__(self):
        curpath = os.path.dirname(os.path.abspath(__file__))
        dirpath = os.path.dirname(curpath)
        filePath = os.path.join(dirpath, 'Log/Error.log')
        self._logFilePath = filePath
        ensurePathExist(os.path.dirname(filePath))

        self._logger = logging.getLogger('error')
        fh = logging.handlers.RotatingFileHandler(filePath, maxBytes=MAX_SIZE, backupCount=10, encoding='utf-8')

        formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
        fh.setFormatter(formatter)

        self._logger.addHandler(fh)
        self._logger.setLevel(logging.DEBUG)

        hostname = socket.gethostname()
        self._logger.info('Host: ' + hostname)
        self._logger.info('Address: ' + socket.gethostbyname(hostname))

    def report(self):
        traceback.print_exc()
        self._logger.error(traceback.format_exc())

        x = list(self._logger.handlers)
        for i in x:
            self._logger.removeHandler(i)
            i.flush()
            i.close()
        self.openLogFile()

    def openLogFile(self):
        if platform.system() == 'Windows':
            os.startfile(self._logFilePath)

    @property
    def logFilePath(self) -> str:
        return self._logFilePath


def loadImageFile(filePath, withDim=False):
    try:
        if os.path.splitext(filePath)[-1] == '.tif':
            numpyarray = cv2.imread(filePath, -1)
        else:
            stream = open(filePath, "rb")
            bytestream = bytearray(stream.read())
            numpyarray = np.asarray(bytestream, dtype=np.uint8)

        imgTemp = cv2.imdecode(numpyarray, cv2.IMREAD_COLOR)
        if imgTemp is not None:
            if np.array_equal(imgTemp[:, :, 0], imgTemp[:, :, 1]) and \
                    np.array_equal(imgTemp[:, :, 1], imgTemp[:, :, 2]):
                dim = 2
            else:
                dim = 3
            imgTemp = cv2.cvtColor(imgTemp, cv2.COLOR_BGR2RGB)
        else:
            imgTemp = numpyarray
            dim = imgTemp.ndim
        if withDim:
            return imgTemp, dim
        else:
            return imgTemp
    except Exception:
        ErrorHandler().report()
        if withDim:
            return None, None
        else:
            return None



if sys.platform != 'win32':
    from time import perf_counter

    try:
        from time import perf_counter_ns
    except ImportError:
        def perf_counter_ns():
            """perf_counter_ns() -> int

            Performance counter for benchmarking as nanoseconds.
            """
            return int(perf_counter() * 10 ** 9)
else:
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel32.QueryPerformanceFrequency.argtypes = (wintypes.PLARGE_INTEGER,)  # lpFrequency
    kernel32.QueryPerformanceCounter.argtypes = (wintypes.PLARGE_INTEGER,)  # lpPerformanceCount
    _qpc_frequency = wintypes.LARGE_INTEGER()
    if not kernel32.QueryPerformanceFrequency(ctypes.byref(_qpc_frequency)):
        raise ctypes.WinError(ctypes.get_last_error())
    _qpc_frequency = _qpc_frequency.value


    def perf_counter_ns():
        """perf_counter_ns() -> int

        Performance counter for benchmarking as nanoseconds.
        """
        count = wintypes.LARGE_INTEGER()
        if not kernel32.QueryPerformanceCounter(ctypes.byref(count)):
            raise ctypes.WinError(ctypes.get_last_error())
        return (count.value * 10 ** 9) // _qpc_frequency


    def perf_counter():
        """perf_counter() -> float

        Performance counter for benchmarking.
        """
        count = wintypes.LARGE_INTEGER()
        if not kernel32.QueryPerformanceCounter(ctypes.byref(count)):
            raise ctypes.WinError(ctypes.get_last_error())
        return count.value / _qpc_frequency
