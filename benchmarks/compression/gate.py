"""The round-trip gate every codec must pass, chunk by chunk.

Deliberately stateful: it keeps the previous chunk's output and reference bytes, so a
decompressor that hands back a view into a buffer it reuses on the next call is caught (the
per-call check alone waved that through — review round cfdb-compression-1). The reference bytes
are snapshotted BEFORE compress, so a compressor that mutates its input in place is caught too.
Comparison is ``tobytes()`` equality: bit-exact, NaN-payload-exact, dtype-agnostic.
``test_gate.py`` holds one mutant per criterion; each must FAIL on its own.
"""


class GateError(AssertionError):
    pass


class RoundTripGate:
    def __init__(self):
        self._prev = None  # (out, ref, label) of the previous chunk

    @staticmethod
    def snapshot(arr):
        """Call BEFORE compress: the bytes the codec must give back."""
        return arr.tobytes()

    def check(self, arr, ref, out, label):
        """``arr`` is the input handed to compress, ``ref`` its snapshot, ``out`` the decompressed
        result. Raises GateError naming the criterion that failed."""
        if arr.tobytes() != ref:
            raise GateError(f'{label}: compress mutated its input')
        if out.dtype != arr.dtype:
            raise GateError(f'{label}: dtype {out.dtype} != {arr.dtype}')
        if out.shape != arr.shape:
            raise GateError(f'{label}: shape {out.shape} != {arr.shape}')
        if not out.flags.c_contiguous:
            raise GateError(f'{label}: output is not C-contiguous')
        if out.tobytes() != ref:
            raise GateError(f'{label}: output bytes differ from input')
        if self._prev is not None:
            pout, pref, plabel = self._prev
            if pout.tobytes() != pref:
                raise GateError(f'{plabel}: earlier output changed after decompressing {label} (shared/reused buffer)')
        self._prev = (out, ref, label)
