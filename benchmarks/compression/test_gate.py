"""One mutant per gate criterion. Run: uv run pytest benchmarks/compression/test_gate.py -q
Each mutant is a codec that is wrong in exactly one way; the gate must refuse each one and pass
the two controls. A mutant that passes means the gate has an escape route."""
import numpy as np
import pytest

from benchmarks.compression.codecs import zstd_codec, shuffle_zstd, shuffle_ydelta_zstd
from benchmarks.compression.gate import RoundTripGate, GateError

rng = np.random.default_rng(0)
CHUNKS = [rng.integers(0, 60000, (3, 1, 20, 17)).astype('uint16'),
          rng.integers(0, 60000, (3, 1, 20, 17)).astype('uint16')]
NAN_CHUNK = rng.random((3, 1, 20, 17)).astype('float32')
NAN_CHUNK[0, 0, 0, :2] = [np.float32('nan'), np.float32('nan')]
NAN_CHUNK.view('uint32')[0, 0, 0, 1] |= 0x1  # second NaN carries a different payload


def run(codec, chunks=CHUNKS):
    comp, decomp = codec
    gate = RoundTripGate()
    for i, a in enumerate(chunks):
        ref = gate.snapshot(a)
        buf = comp(a)
        out = decomp(buf, str(a.dtype), a.shape)
        gate.check(a, ref, out, f'chunk{i}')


@pytest.mark.parametrize('codec', [zstd_codec(1), shuffle_zstd(1), shuffle_ydelta_zstd(1)])
def test_ok_controls(codec):
    run(codec)
    run(codec, [NAN_CHUNK, NAN_CHUNK.copy()])


def mutant_flip_byte():
    c, d = zstd_codec(1)
    def decomp(b, dt, sh):
        o = d(b, dt, sh).copy(); o.reshape(-1).view('u1')[5] ^= 0xFF; return o
    return c, decomp


def mutant_truncated_plane():
    """shuffle codec that drops the high byte plane: half the bytes, every value wrong"""
    c, d = shuffle_zstd(1)
    def decomp(b, dt, sh):
        o = d(b, dt, sh).copy(); o >>= 8; return o
    return c, decomp


def mutant_wrong_dtype():
    c, d = zstd_codec(1)
    return c, lambda b, dt, sh: d(b, dt, sh).view('int16')


def mutant_wrong_shape():
    c, d = zstd_codec(1)
    return c, lambda b, dt, sh: d(b, dt, sh).reshape(-1)


def mutant_input_mutation():
    """compressor XORs its input in place, then compresses the mutated array (Fable's mutant)"""
    c, d = zstd_codec(1)
    def comp(a):
        if a.flags.writeable:
            a ^= 1
        return c(a)
    return comp, d


def mutant_shared_buffer():
    """decompressor returns a view into one reused buffer (Sonnet's/Gemini's mutant)"""
    c, d = zstd_codec(1)
    state = {}
    def decomp(b, dt, sh):
        o = d(b, dt, sh)
        if 'buf' not in state:
            state['buf'] = np.empty(o.shape, o.dtype)
        state['buf'][...] = o
        return state['buf']
    return c, decomp


def mutant_fortran_output():
    c, d = zstd_codec(1)
    return c, lambda b, dt, sh: np.asfortranarray(d(b, dt, sh))


def mutant_nan_canonicalised():
    """decompressor rewrites every NaN to the canonical NaN: values 'equal' but bits differ"""
    c, d = zstd_codec(1)
    def decomp(b, dt, sh):
        o = d(b, dt, sh).copy(); o[np.isnan(o)] = np.float32('nan'); return o
    return c, decomp


@pytest.mark.parametrize('mutant', [mutant_flip_byte, mutant_truncated_plane, mutant_wrong_dtype, mutant_wrong_shape,
                                    mutant_input_mutation, mutant_shared_buffer, mutant_fortran_output])
def test_mutants_refused(mutant):
    with pytest.raises(GateError):
        run(mutant())


def test_nan_payload_refused():
    with pytest.raises(GateError):
        run(mutant_nan_canonicalised(), [NAN_CHUNK])


def test_nan_payload_would_pass_array_equal():
    """documents why the gate is tobytes() and not array_equal(equal_nan=True)"""
    c, d = mutant_nan_canonicalised()
    out = d(c(NAN_CHUNK), 'float32', NAN_CHUNK.shape)
    assert np.array_equal(out, NAN_CHUNK, equal_nan=True)
    assert out.tobytes() != NAN_CHUNK.tobytes()
