from __future__ import annotations


from enum import Enum
from functools import partial
from time import sleep
from typing import TYPE_CHECKING

import numpy as np
import pyvisa

import skrf
from skrf.frequency import Frequency
from skrf.vi.vna import VNA

if TYPE_CHECKING:
    from typing import Union


# command byte and length of expected data
class OP(tuple[bytes, int], Enum):
    NOP = b"\x00", 0
    INDICATE = b"\x0d", 0
    READ = b"\x10", 1
    READ2 = b"\x11", 2
    READ4 = b"\x12", 4
    READFIFO = b"\x18", -1
    WRITE = b"\x20", 1
    WRITE2 = b"\x21", 2
    WRITE4 = b"\x22", 4
    WRITE8 = b"\x23", 8
    WRITEFIFO = b"\x28", -1


class RegAddr(bytes, Enum):
    SWEEP_START = b"\x00"
    SWEEP_STEP = b"\x10"
    SWEEP_POINTS = b"\x20"
    VALS_PER_FREQ = b"\x22"
    RAW_SAMPLES_MODE = b"\x26"
    VALS_FIFO = b"\x30"
    DEVICE_VARIANT = b"\xf0"
    PROTOCOL_VERSION = b"\xf1"
    HARDWARE_REV = b"\xf2"
    FIRMWARE_MAJOR = b"\xf3"
    FIRMWARE_MINOR = b"\xf4"


from_bytes = partial(int.from_bytes, byteorder="little", signed=True)
from_ubytes = partial(int.from_bytes, byteorder="little", signed=False)
to_ubytes = partial(int.to_bytes, byteorder="little", signed=False)
make_freq = partial(Frequency, unit="hz", sweep_type="lin")


def _convert_bytes_to_sparams(
    count: int, raw: bytearray
) -> tuple[np.ndarray, np.ndarray]:
    s11 = np.zeros(count, dtype=complex)
    s21 = np.zeros_like(s11)

    for i in range(count):
        start = i * 32
        stop = (i + 1) * 32
        chunk = raw[start:stop]
        fwd0re = from_bytes(chunk[:4])
        fwd0im = from_bytes(chunk[4:8])
        rev0re = from_bytes(chunk[8:12])
        rev0im = from_bytes(chunk[12:16])
        rev1re = from_bytes(chunk[16:20])
        rev1im = from_bytes(chunk[20:24])
        freq_index = from_bytes(chunk[24:26])

        fwd0 = complex(fwd0re, fwd0im)
        rev0 = complex(rev0re, rev0im)
        rev1 = complex(rev1re, rev1im)

        s11[freq_index] = rev0 / fwd0
        s21[freq_index] = rev1 / fwd0

    return s11, s21


class NanoVNAv2(VNA):
    _scpi = False

    def __init__(self, address, backend: str = "@py"):
        super().__init__(address, backend)
        if not isinstance(self._resource, pyvisa.resources.SerialInstrument):
            raise RuntimeError(
                "NanoVNA_V2 can only be a serial instrument."
                f"{address} yields a {self._resource.__class__.__name__}"
            )

        self.read_bytes = self._resource.read_bytes
        self.write_raw = self._resource.write_raw
        self.query_delay = self._resource.query_delay = 0.05
        self._freq = make_freq(start=100e6, stop=300e6, npoints=201)
        self._reset_protocol()
        self._set_sweep()

    def _reset_protocol(self) -> None:
        self.write_raw(OP.NOP[0] * 8)
        sleep(self.query_delay)

    def query(self, cmd: OP, addr: RegAddr, count: int = 0, bytesize: int = 1) -> bytes:
        bytes_to_read = cmd[1]
        if cmd == OP.READFIFO:
            self.write_raw(cmd[0] + addr + to_ubytes(count, 1))
            bytes_to_read = count * bytesize
        else:
            self.write_raw(cmd[0] + addr)
        sleep(self.query_delay)
        return self.read_bytes(bytes_to_read)

    def write(self, cmd: OP, addr: RegAddr, data: Union[bytes, int] = None) -> None:
        if cmd == OP.WRITEFIFO:
            self.write_raw(cmd[0] + addr + to_ubytes(len(data)) + data)
            return
        if cmd[1] == 0:  # command without data
            self.write_raw(cmd[0] + addr)
            return
        if isinstance(data, (int, float)):
            data = to_ubytes(int(data), cmd[1])
        self.write_raw(cmd[0] + addr + data)

    def _device_info_raw(self) -> tuple[int, int, int, int, int]:
        return (
            from_ubytes(self.query(OP.READ, RegAddr.DEVICE_VARIANT, 1)),
            from_ubytes(self.query(OP.READ, RegAddr.PROTOCOL_VERSION, 1)),
            from_ubytes(self.query(OP.READ, RegAddr.HARDWARE_REV, 1)),
            from_ubytes(self.query(OP.READ, RegAddr.FIRMWARE_MAJOR, 1)),
            from_ubytes(self.query(OP.READ, RegAddr.FIRMWARE_MINOR, 1)),
        )

    def _set_sweep(self) -> None:
        self.write(OP.WRITE8, RegAddr.SWEEP_START, self._freq.start)
        self.write(OP.WRITE8, RegAddr.SWEEP_STEP, self._freq.step)
        self.write(OP.WRITE2, RegAddr.SWEEP_POINTS, self._freq.npoints)

    @property
    def id(self) -> str:
        device, _, hardware, fw_major, fw_minor = self._device_info_raw()
        if device != 2:
            return f"Unknown device, got deviceVariant={device}"
        return f"NanoVNA_V2_{hardware} FW({fw_major}.{fw_minor})"

    @property
    def device_info(self) -> str:
        device, protocol, hardware, fw_major, fw_minor = self._device_info_raw()
        return (
            f"NanoVNA V2\n"
            f"\tVariant:{device}\n"
            f"\tProtocol Version:{protocol}\n"
            f"\tHardware Version: {hardware}\n"
            f"\tFirmware Version: {fw_major}.{fw_minor}"
        )

    @property
    def freq_start(self) -> float:
        return self._freq.start

    @freq_start.setter
    def freq_start(self, freq: float) -> None:
        self._freq = make_freq(freq, self._freq.stop, self._freq.npoints)
        self._set_sweep()

    @property
    def freq_stop(self) -> float:
        return self._freq.stop

    @freq_stop.setter
    def freq_stop(self, freq: float) -> None:
        self._freq = make_freq(self._freq.start, freq, self._freq.npoints)
        self._set_sweep()

    @property
    def freq_step(self) -> float:
        return self._freq.step

    @freq_step.setter
    def freq_step(self, freq: float) -> None:
        self._freq = Frequency.from_f(
            range(
                self._freq.start,
                self._freq.start + freq * (self._freq.npoints + 1),
                freq,
            ),
            unit="hz",
        )
        self._set_sweep()

    @property
    def npoints(self) -> int:
        return self._freq.npoints

    @npoints.setter
    def npoints(self, count: int) -> None:
        self._freq = make_freq(self._freq.start, self._freq.stop, count)
        self._set_sweep()

    @property
    def frequency(self) -> Frequency:
        return self._freq

    @frequency.setter
    def frequency(self, freq: Frequency):
        self._freq = freq
        self._set_sweep()

    def clear_fifo(self) -> None:
        self.write(OP.WRITE, RegAddr.VALS_FIFO, 0)

    def get_s11_s21(self) -> tuple[skrf.Network, skrf.Network]:
        points = self._freq.npoints
        self.clear_fifo()
        sleep(self.query_delay)

        raw = bytearray()
        points_remaining = points

        while points_remaining > 0:
            len_segment = min(points_remaining, 255)
            data = self.query(OP.READFIFO, RegAddr.VALS_FIFO, len_segment, 32)
            raw.extend(data)
            points_remaining = points_remaining - len_segment

        s11, s21 = skrf.Network(), skrf.Network()
        s11.frequency = self._freq.copy()
        s21.frequency = self._freq.copy()

        s11.s, s21.s = _convert_bytes_to_sparams(points, raw)

        return s11, s21
