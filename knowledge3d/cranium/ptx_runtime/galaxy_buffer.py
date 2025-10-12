import ctypes

class GalaxyEmbedding(ctypes.Structure):
    _fields_ = [
        ("vector", ctypes.c_float * 4),
        ("metadata", ctypes.c_uint32),
        ("galaxy_clock", ctypes.c_uint16),
        ("access_freq", ctypes.c_uint16),
        ("checksum", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32)
    ]

GALAXY_EMBEDDING_SIZE = ctypes.sizeof(GalaxyEmbedding)
GALAXY_CLOCK_MAX = 65535
