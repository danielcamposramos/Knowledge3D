from __future__ import annotations

import ctypes

from knowledge3d.cranium.sovereign.loader import _coerce_host_pointer


def test_coerce_host_pointer_accepts_byref_objects():
    value = ctypes.c_uint32(7)
    pointer = _coerce_host_pointer(ctypes.byref(value))

    assert isinstance(pointer, ctypes.c_void_p)
    assert int(pointer.value) == ctypes.addressof(value)
