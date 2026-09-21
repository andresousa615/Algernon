# -*- coding: utf-8 -*-
"""
Lightweight NVTX wrapper for processes without a CUDA context (loader workers).

Calls libnvToolsExt directly through ctypes, so it needs neither torch.cuda
nor an active CUDA context and works in forked worker processes. When the
library is missing every call is a no-op.

Usage:
    from profiling.worker_nvtx import push, pop, range_ctx

    push("algernon/aug/io"); ...; pop()

    with range_ctx("algernon/aug/spatial"):
        ...
"""

import contextlib
import ctypes

_lib = None


def _get_lib():
    global _lib
    if _lib is None:
        try:
            lib = ctypes.CDLL("libnvToolsExt.so.1")
            # On nodes without CUDA the library may be a stub without the symbols.
            _ = lib.nvtxRangePushA
            _ = lib.nvtxRangePopA
            _lib = lib
        except (OSError, AttributeError):
            _lib = False   # missing or stub — calls become no-ops
    return _lib


def push(name: str) -> None:
    lib = _get_lib()
    if lib:
        lib.nvtxRangePushA(ctypes.c_char_p(name.encode()))


def pop() -> None:
    lib = _get_lib()
    if lib:
        lib.nvtxRangePopA()


@contextlib.contextmanager
def range_ctx(name: str):
    push(name)
    try:
        yield
    finally:
        pop()
