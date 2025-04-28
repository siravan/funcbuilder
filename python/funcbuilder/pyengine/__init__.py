import sys
import ctypes
import mmap
from ctypes.util import find_library
import platform

from . import amd
from . import arm
from . import tree


def can_compile():
    return (sys.platform in ["linux", "darwin"] and arch() in ["amd", "arm"]) or (
        sys.platform == "win32" and arch() == "amd"
    )


def arch():
    if platform.machine() in ["x86_64", "AMD64"]:
        return "amd"
    elif platform.machine() in ["arm64", "aarch64"]:
        return "arm"
    else:
        return None


class Memory:
    COUNT_SPILLS = 16
    ZERO = 0
    ONE = 1
    MINUS_ONE = 2
    MINUS_ZERO = 3

    def __init__(self, model):
        self.consts = [0.0, 1.0, -1.0, -0.0]
        self.names = []

        self.first_state = len(self.names)
        self.count_states = len(model.states)

        for var in model.states:
            self.names.append(var.name)

        self.first_spill = len(self.names)
        for i in range(Memory.COUNT_SPILLS):
            self.names.append(f"#{i}")

        self.first_obs = len(self.names)
        self.count_obs = len(model.obs)

        for var in model.obs:
            self.names.append(var.name)

    def index(self, name):
        if isinstance(name, tree.Var):
            name = name.name
        return self.names.index(name)

    def add_index(self, name):
        if isinstance(name, tree.Var):
            name = name.name

        if name in self.names:
            return self.names.index(name)

        self.names.append(name)
        self.count_obs += 1
        return len(self.names) - 1

    def constant(self, val):
        val = float(val)
        try:
            return self.consts.index(val)
        except:
            self.consts.append(val)
            return len(self.consts) - 1

    def spill(self, reg):
        assert reg < Memory.COUNT_SPILLS
        return self.first_spill + reg


class VirtualTable:
    libm = None

    def __init__(self):
        if VirtualTable.libm == None:
            if sys.platform == "win32":
                VirtualTable.libm = ctypes.cdll.msvcrt
            else:
                path = find_library("m")
                VirtualTable.libm = ctypes.CDLL(path)

        self.addr = []
        self.dict = {}
        libm = VirtualTable.libm

        self.exp = self.unary(libm.exp, "exp")
        self.log = self.unary(libm.log, "log")
        self.log10 = self.unary(libm.log10, "log10")
        self.pow = self.binary(libm.pow, "pow")

        self.sin = self.unary(libm.sin, "sin")
        self.cos = self.unary(libm.cos, "cos")
        self.tan = self.unary(libm.tan, "tan")

        self.sinh = self.unary(libm.sinh, "sinh")
        self.cosh = self.unary(libm.cosh, "cosh")
        self.tanh = self.unary(libm.tanh, "tanh")

        self.asin = self.unary(libm.asin, "asin")
        self.acos = self.unary(libm.acos, "acos")
        self.atan = self.unary(libm.atan, "atan")

        # msvcrt does not export inverse hyperbolic functions!
        if sys.platform != "win32":
            self.asinh = self.unary(libm.asinh, "asinh")
            self.acosh = self.unary(libm.acosh, "acosh")
            self.atanh = self.unary(libm.atanh, "atanh")

    def unary(self, func, name):
        self.addr.append(ctypes.c_ulonglong.from_buffer(func).value)
        self.dict[name] = len(self.addr) - 1
        func.restype = ctypes.c_double
        func.argtypes = [ctypes.c_double]
        return func

    def binary(self, func, name):
        self.addr.append(ctypes.c_ulonglong.from_buffer(func).value)
        self.dict[name] = len(self.addr) - 1
        func.restype = ctypes.c_double
        func.argtypes = [ctypes.c_double, ctypes.c_double]
        return func

    def vt(self):
        return self.addr

    def find(self, op):
        return self.dict[op]


class Code:
    alloc = None
    free = None
    libc = None
    mprotect = None
    icache_invalidate = None

    MEM_COMMIT = 0x00001000
    MEM_RESERVE = 0x00002000
    PAGE_EXECUTE_READ = 0x00000020
    PAGE_EXECUTE_READWRITE = 0x00000040
    PAGE_READWRITE = 0x00000004
    MEM_RELEASE = 0x00008000
    MEM_DECOMMIT = 0x00004000

    def __init__(self, buf):
        if sys.platform == "win32":
            if Code.alloc is None:
                Code.alloc = ctypes.windll.kernel32.VirtualAlloc
                Code.alloc.restype = ctypes.c_void_p
                Code.alloc.argtypes = [
                    ctypes.c_void_p,  # lpAddress
                    ctypes.c_size_t,  # dwSize
                    ctypes.c_uint32,  # flAllocationType
                    ctypes.c_uint32,  # flProtect
                ]

                Code.free = ctypes.windll.kernel32.VirtualFree
                Code.free.restype = ctypes.c_bool
                Code.free.argtypes = [
                    ctypes.c_void_p,  # lpAddress
                    ctypes.c_size_t,  # dwSize
                    ctypes.c_uint32,  # dwFreeType
                ]

            size = (1 + len(buf) // 4096) * 4096
            self.addr = Code.alloc(
                None,
                size,
                Code.MEM_RESERVE | Code.MEM_COMMIT,
                Code.PAGE_EXECUTE_READWRITE,
            )
            buf = bytes(buf)
            ctypes.memmove(self.addr, buf, len(buf))
        else:  # linux/darwin
            # macOS, especially on Apple Silicon, requires special care!
            # 1. We need to pass MAP_JIT to mmap
            # 2. W^X rule: the memory cannot be both writable and executable
            # 3. We create the memory as Read+Write, write the code, and then
            #   change the mode to Read+Execute using mprotect
            # 4. Question: do we need to call pthread_jit_write_protect_np?
            # 5. Question: do we need to invalidate cache on aarch64?
            #   note: function invalidate_cache is present but not called yet
            try:
                MAP_JIT = mmap.MAP_JIT  # mmap.MAP_JIT is defined in python 1.13
            except:
                MAP_JIT = 0x00000800

            self.code = mmap.mmap(
                -1,
                len(buf),
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
                flags=mmap.MAP_ANON | mmap.MAP_PRIVATE | MAP_JIT,
            )

            self.code.write(buf)
            self.code.flush()
            self.addr = ctypes.addressof(ctypes.c_long.from_buffer(self.code))

            if Code.mprotect == None:
                Code.mprotect = ctypes.pythonapi.mprotect
                Code.mprotect.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_size_t,
                    ctypes.c_int,
                ]
                Code.mprotect.restype = ctypes.c_int

            Code.mprotect(self.addr, len(buf), mmap.PROT_READ | mmap.PROT_EXEC)

    def invalidate_cache(self):
        if Code.libc == None:
            path = find_library("c")
            Code.libc = ctypes.CDLL(path)

        if Code.icache_invalidate == None:
            Code.icache_invalidate = self.libc.sys_icache_invalidate
            Code.icache_invalidate.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            Code.icache_invalidate.restype = ctypes.c_int
        Code.icache_invalidate(self.addr, len(self.code))

    def __del__(self):
        if sys.platform == "win32":
            Code.free(self.addr, 0, Code.MEM_RELEASE)
            # note: for linux, mmap takes care of releasing memory


class PyCompiler:
    def __init__(self, model, y, ty="native", sig=None):
        self.mem = Memory(model)
        self.vt = VirtualTable()
        self.prog = self.assembler(ty)(self.mem)
        
        model.compile(y, self.prog, self.mem, self.vt)

        if sig == None:
            sig = [ctypes.c_double for _ in range(self.mem.count_states + 1)]

        fac = ctypes.CFUNCTYPE(*sig)

        self.code = Code(self.prog.buf())
        self.func = fac(self.code.addr)

        self.populate()

    def __del__(self):
        pass
        # free_function(self.addr)
        # free_function(self.vaddr)

    def assembler(self, ty):
        a = arch()

        if a == "amd" and (ty == "native" or ty == "amd"):
            prog = amd.AmdIR
            self.can_run = True
        elif a == "arm" and (ty == "native" or ty == "arm"):
            prog = arm.ArmIR
            self.can_run = True
        elif a == "amd" and ty == "arm":
            prog = arm.ArmIR
            self.can_run = False
            print(
                "x64 code is requested on a non-compatible platform. Cannot run the resulting code."
            )
        elif a == "arm" and ty == "amd":
            prog = amd.AmdIR
            self.can_run = False
            print(
                "aarch64 code is requested on a non-compatible platform. Cannot run the resulting code."
            )
        else:
            raise ValueError(f"unrecognized processor type: {ty}")
        return prog

    def populate(self):
        first_state = self.mem.first_state
        self.first_state = first_state
        first_obs = self.mem.first_obs
        self.count_states = self.mem.count_states
        self.count_obs = self.mem.count_obs

    def dumps(self):
        return self.prog.buf().hex()

    def dump(self, name=None):
        with open(name, "wb") as fd:
            fd.write(prog.buf)
