import os
import sys
import time
import queue
import serial
import serial.rs485
from serial.tools import list_ports, list_ports_common
import threading
from typing import List, Union

FILE_PATH = os.path.dirname(os.path.realpath(__file__))  # %PROJECT_ROOT%/Include
ROOT_PATH = os.path.dirname(os.path.dirname(FILE_PATH))
INCLUDE_PATH = os.path.join(ROOT_PATH, "Include")
RESOURCES_PATH = os.path.join(ROOT_PATH, "Resources")
LIBRARY_PATH = os.path.join(ROOT_PATH, "Library")
SERIAL_PATH = os.path.join(LIBRARY_PATH, "Serial")
sys.path.extend([INCLUDE_PATH, RESOURCES_PATH])
sys.path = list(set(sys.path))
del FILE_PATH, ROOT_PATH, INCLUDE_PATH, RESOURCES_PATH, LIBRARY_PATH, SERIAL_PATH

from Commons import PySignal, is_system_win, write_log


def get_serial_port_list() -> List[list_ports_common.ListPortInfo]:
    return sorted(list_ports.comports())


def get_serial_port_name_list() -> List[str]:
    list_port_name = []
    for com_port in get_serial_port_list():
        list_port_name.append(com_port.device)

    return list_port_name


def get_serial_baud_list() -> List[str]:
    return ["2400", "4800", "9600", "19200", "28800",
            "38400", "57600", "76800", "115200", "230400", "460800", "576000", "921600"]


def get_serial_port_description(astrDevice: str) -> str:
    list_serial_port = get_serial_port_list()

    for port in list_serial_port:
        if astrDevice in port.device:
            return str(port.description.split("(")[0])


class CSerialPort(object):
    threadSendRecvDirect = None
    threadSend = None
    threadRecv = None
    threadRecvQueue = None
    btaLastSendData = None
    btaLastRecvData = None
    bIsRecvDirect: bool

    def __init__(self, abIsRecvDirect=False):
        super().__init__()
        self.sig_connected = PySignal()
        self.sig_disconnected = PySignal()
        self.sig_send_data = PySignal(bytes)
        self.sig_recv_data = PySignal(bytes)
        self.sig_serial_error = PySignal()

        self._serial = serial.Serial()  # port=None : do not try to connect
        self.bIsRecvDirect = abIsRecvDirect
        self._sendQueue = queue.Queue()
        self._recvQueue = queue.Queue()

    def setParams(self, **kwargs):
        try:
            if 'port' in kwargs.keys():
                self._serial.port = kwargs['port']
            if 'baudrate' in kwargs.keys():
                self._serial.baudrate = kwargs['baudrate']
            if 'bytesize' in kwargs.keys():
                self._serial.bytesize = kwargs['bytesize']
            if 'parity' in kwargs.keys():
                self._serial.parity = kwargs['parity']
            if 'stopbits' in kwargs.keys():
                self._serial.stopbits = kwargs['stopbits']
            if 'timeout' in kwargs.keys():
                self._serial.timeout = kwargs['timeout']
                self._serial.write_timeout = kwargs['timeout']
        except Exception:
            pass  # todo: add exception handler

    def get_port(self):
        return self._serial.port

    def get_baudrate(self):
        return self._serial.baudrate

    def connect(self):
        try:
            if self._serial.isOpen():
                return

            self._serial.open()
            if self._serial.isOpen():
                self.clearQueue()
                self.start_threads()
                self.sig_connected.emit()
        except serial.SerialException as e:
            pass  # todo: add exception handler

        except Exception:
            pass  # todo: add exception handler

    def is_connected(self):
        try:
            return self._serial.isOpen()
        except Exception:
            pass  # todo: add exception handler

    def disconnect(self):
        try:
            if self._serial.isOpen():
                self._serial.cancel_read()  # cancel read_until(etx)
                self.stop_threads()
                self._serial.close()
                self.sig_disconnected.emit()
        except Exception:
            pass  # todo: add exception handler

    def start_threads(self):
        try:

            if self.bIsRecvDirect is False:
                if self.threadSend is None:
                    self.threadSend = SendThread(s=self._serial, q=self._sendQueue)
                    self.threadSend.sig_send.connect(self.onDataSend)
                    self.threadSend.start()

                if self.threadRecv is None:
                    self.threadRecv = ReceiveThread(q=self._recvQueue)
                    self.threadRecv.sig_recv.connect(self.onDataReceive)
                    self.threadRecv.start()

                if self.threadRecvQueue is None:
                    self.threadRecvQueue = RecvQueueThread(s=self._serial, q=self._recvQueue)
                    self.threadRecvQueue.start()
            else:
                if self.threadSendRecvDirect is None:
                    self.threadSendRecvDirect = SendRecvDirectThread(s=self._serial, q=self._sendQueue)
                    self.threadSendRecvDirect.sig_send.connect(self.onDataSend)
                    self.threadSendRecvDirect.sig_recv.connect(self.onDataReceive)
                    self.threadSendRecvDirect.sig_error.connect(self.onDataError)
                    self.threadSendRecvDirect.start()

        except Exception:
            pass  # todo: add exception handler

    def stop_threads(self):
        try:
            if self.threadRecvQueue is not None:
                self.threadRecvQueue._keepAlive = False
                self.threadRecvQueue.stop()
                while self.threadRecvQueue.is_alive():
                    pass
                self.threadRecvQueue = None
            if self.threadRecv is not None:
                self.threadRecv._keepAlive = False
                while self.threadRecv.is_alive():
                    pass
                self.threadRecv = None
            if self.threadSend is not None:
                self.threadSend._keepAlive = False
                while self.threadSend.is_alive():
                    pass
                self.threadSend = None

            if self.threadSendRecvDirect is not None:
                self.threadSendRecvDirect._keepAlive = False
                while self.threadSendRecvDirect.is_alive():
                    pass
                self.threadSendRecvDirect = None

        except Exception:
            pass  # todo: add exception handler

    def onDataError(self):
        self.sig_serial_error.emit()

    # convert to bytes. parsing should be done on the caller as a callback.
    def onDataSend(self, data):
        try:
            self.btaLastSendData = data
            self.sig_send_data.emit(data)
        except Exception:
            pass  # todo: add exception handler

    # convert to bytes. parsing should be done on the caller as a callback.
    def onDataReceive(self, data):
        try:
            self.btaLastRecvData = data
            self.sig_recv_data.emit(data)
        except Exception:
            pass  # todo: add exception handler

    def clearQueue(self):
        try:
            while not self._sendQueue.empty():
                self._sendQueue.get(block=True)
            while not self._recvQueue.empty():
                self._recvQueue.get(block=True)
        except Exception:
            pass  # todo: add exception handler

    def sendData(self, data, abIsWaitFeedback: bool = False):
        try:
            sData = bytes()
            if isinstance(data, str):  # String to Bytes: Ascii
                tmp = bytearray()
                tmp.extend(map(ord, data))
                sData = bytes(tmp)
            elif isinstance(data, bytes) or isinstance(data, bytearray):
                sData = bytes(data)

            if self.bIsRecvDirect:
                self._sendQueue.put((sData, abIsWaitFeedback))
            else:
                self._sendQueue.put(sData)

        except Exception:
            pass  # todo: add exception handler

    def get_last_send_data(self):
        return self.btaLastSendData

    def get_last_recv_data(self):
        return self.btaLastRecvData


class SendRecvDirectThread(threading.Thread):
    def __init__(self, s: serial.Serial, q: queue.Queue, etx: Union[bytearray, None] = None):
        super(SendRecvDirectThread, self).__init__()
        self.setDaemon(True)
        self.sig_send = PySignal(bytes)
        self.sig_recv = PySignal(bytes)
        self.sig_error = PySignal()
        self._keepAlive = True
        self._serial = s
        self._queue = q

        # todo: process if there is no ETX
        self._etx = bytearray([0x03])

    def stop(self):
        self._keepAlive = False
        self._queue.put(("STOP_RECV_DIRECT_THREAD", False))

    def run(self) -> None:
        write_log("Terminate thread", self)
        while self._keepAlive is True:
            try:
                data, b_feedback = self._queue.get(block=True)
                if data == "STOP_RECV_DIRECT_THREAD":
                    break
                else:
                    sendLen = len(data)
                    while sendLen > 0:
                        nLen = self._serial.write(data[(len(data) - sendLen):])
                        sData = data[(len(data) - sendLen):(len(data) - sendLen + nLen)]
                        self.sig_send.emit(sData)
                        sendLen -= nLen

                    if b_feedback:
                        feedback = self._serial.read_until(self._etx)
                        self.sig_recv.emit(feedback)

                time.sleep(1e-6)
            except Exception:
                self.sig_error.emit()

        write_log("Terminate thread", self)


class SendThread(threading.Thread):
    def __init__(self, s, q: queue.Queue):
        super(SendThread, self).__init__()
        self.setDaemon(True)
        self.sig_send = PySignal(bytes)
        self._keepAlive = True
        self._serial = s
        self._queue = q

    def stop(self):
        self._keepAlive = False
        self._queue.put("STOP_SEND_THREAD")

    def run(self):
        write_log("Start thread", self)
        while self._keepAlive is True:
            try:
                data = self._queue.get(block=True)
                if data == "STOP_SEND_THREAD":
                    break
                else:
                    sendLen = len(data)
                    while sendLen > 0:
                        nLen = self._serial.write(data[(len(data) - sendLen):])
                        sData = data[(len(data) - sendLen):(len(data) - sendLen + nLen)]
                        self.sig_send.emit(sData)
                        sendLen -= nLen

                time.sleep(1e-6)
            except Exception:
                pass  # todo: add exception handler

        write_log("Terminate thread", self)


# utilize queue to store received data
class ReceiveThread(threading.Thread):
    def __init__(self, q):
        threading.Thread.__init__(self)
        self.setDaemon(True)
        self.sig_recv = PySignal(bytes)
        self._keepAlive = True
        self._queue = q

    def stop(self):
        self._keepAlive = False
        self._queue.put("STOP_RECV_THREAD")

    def run(self):
        write_log("Start thread", self)
        while self._keepAlive is True:
            try:
                data = self._queue.get(block=True)
                if data == "STOP_RECV_THREAD":
                    break
                else:
                    self.sig_recv.emit(data)

                time.sleep(1e-6)

            except Exception:
                pass  # todo: add exception handler

        write_log("Terminate thread", self)


class RecvQueueThread(threading.Thread):
    def __init__(self, s, q):
        threading.Thread.__init__(self)
        self.setDaemon(True)
        self._keepAlive = True
        self._serial = s
        self._queue = q

    def stop(self):
        self._keepAlive = False
        self._queue.put("STOP_QUEUE_THREAD")

    def run(self):
        # todo: add logger (thread started)
        write_log("Start thread", self)
        while self._keepAlive is True:
            try:
                self._serial.timeout = 1
                rcvData = self._serial.read_all()
                # etx = bytearray([0x3E, 0x03])
                # rcvData = self._serial.read_until(etx)
                if len(rcvData) > 0:
                    self._queue.put(rcvData)
                else:
                    time.sleep(1e-6)
            except serial.serialutil.SerialException:
                pass
            except Exception:
                pass  # todo: add exception handler

        write_log("Terminate thread", self)
