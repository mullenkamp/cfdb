import copy
import pytest
import numpy as np
import shapely
from cfdb import dtypes

###################################################
### Test data

point_data = np.array([shapely.Point(1.3, 2.6), shapely.Point(2.3, 3.6)], object)
line_data = np.array([shapely.LineString([(1.3, 2.6), (2.3, 3.6)]), shapely.LineString([(8.6, 6.6), (4.3, 7.6)])], object)
poly_data = np.array([shapely.Polygon([(1.3, 2.6), (2.3, 3.6), (1, 0)]), shapely.Polygon([(8.6, 6.6), (4.3, 7.6), (1, 0)])], object)

float_data = np.linspace(0, 19.9, 200, dtype='float64')
float_data2 = np.linspace(0, 19.9, 200, dtype='float32')
time_data = np.linspace(0, 199, 200, dtype='datetime64[D]')
time_data2 = np.linspace(0, 199, 200, dtype='datetime64[h]')
int_data = np.arange(4, 13, dtype='int16')
int_data2 = np.arange(4, 13)
bool_data = np.zeros((9,), 'bool')
bool_data[4] = True

str_data = np.array(['hello', 'egggggsssss'], np.dtypes.StringDType())

meta = {
    'point': {'precision': 1},
    'linestring': {'precision': 1},
    'polygon': {'precision': 1},
    'float64': {'precision': 1, 'min_value': 0, 'max_value': 20},
    'float32': {'precision': 1},
    'datetime64[D]': {},
    'datetime64[h]': {'min_value': '1970-01-01', 'max_value': '1970-01-11'},
    'bool': {},
    'int16': {},
    'int64': {'min_value': 0, 'max_value': 20},
    'str': {},
    }

test_values = {
    'point': shapely.Point(1.0, 2.0),
    'linestring': shapely.LineString([(1.0, 2.0), (2.6, 3.8)]),
    'polygon': shapely.Polygon([(1.0, 2.0), (2.8, 3.1), (1.8, 0)]),
    'float64': 10.6,
    'float32': 100.6,
    'datetime64[D]': np.array(201, dtype='datetime64[D]'),
    'datetime64[h]': np.array(201, dtype='datetime64[h]'),
    'bool': True,
    'int16': 14,
    'int64': 14,
    'str': 'big string',
    }


##############################
### Functions


##############################
### Tests

@pytest.mark.parametrize('data', [point_data, line_data, poly_data, float_data, float_data2, time_data, time_data2, int_data, int_data2, bool_data, str_data])
def test_dtypes(data):
    np_dtype = data.dtype
    if np_dtype.name == 'object':
        name = data[0].geom_type.lower()
    elif np_dtype.name == 'StringDType128':
        name = 'str'
    else:
        name = np_dtype.name

    dtype1 = dtypes.dtype(name, **meta[name])

    data_b = dtype1.dumps(data)

    assert isinstance(data_b, bytes)

    new_data = dtype1.loads(data_b)

    if 'float' in name:
        assert np.allclose(new_data, data)
    else:
        assert (new_data == data).all()

    test_value = test_values[name]
    new_data[0] = test_value

    data_b = dtype1.dumps(new_data)
    new_data2 = dtype1.loads(data_b)

    if 'float' in name:
        assert np.isclose(new_data2[0], test_value)
    else:
        assert new_data[0] == test_value



















@pytest.mark.parametrize('n', [2, 8, 64])
@pytest.mark.parametrize(('precision', 'min_value', 'max_value'), [
    (4, 0, 100000),   # -> uint32 (the SIMD NaN cast fabricated 214747.36 here pre-fix)
    (3, -5, 55),      # -> uint16
    (1, 0, 100),      # -> uint16 (small range)
    ])
def test_encode_nan_and_inf_roundtrip_to_nan(n, precision, min_value, max_value):
    # NaN/inf written through a packed float dtype must decode back as NaN — never a
    # fabricated value. Parametrized over array sizes because the pre-fix failure was an
    # undefined float->uint cast whose result DIFFERED between numpy's scalar path (small
    # arrays) and SIMD path (larger arrays).
    dtype1 = dtypes.dtype('float32', precision=precision, min_value=min_value, max_value=max_value)

    data = np.full(n, np.nan, dtype='float64')
    data[0] = float(min_value)
    if n > 2:
        data[1] = np.inf
        data[2] = -np.inf

    new_data = dtype1.loads(dtype1.dumps(data), (n,))
    assert np.isclose(new_data[0], min_value)
    assert np.isnan(new_data[1:]).all()


def test_encode_out_of_range_roundtrips_to_nan():
    # Finite values outside the encodable range must decode as NaN (the documented
    # behavior) — never wrap modulo the integer width into plausible in-range values
    # (pre-fix: 60.0 on a [-5, 55] uint16 encoding read back as -5.536).
    dtype1 = dtypes.dtype('float32', precision=3, min_value=-5, max_value=55)

    data = np.array([54.0, 60.0, 65.535, 1e12, -5.0, -7.0, -1e12, 55.0])
    new_data = dtype1.loads(dtype1.dumps(data), (8,))

    np.testing.assert_allclose(new_data[[0, 4, 7]], [54.0, -5.0, 55.0], atol=1e-3)  # in-range exact
    assert np.isnan(new_data[[1, 2, 3, 5, 6]]).all()  # everything unencodable -> NaN


def test_encode_nat_roundtrips_to_nat():
    # NaT through a packed datetime dtype must decode back as NaT, not a fabricated date.
    dtype1 = dtypes.dtype('datetime64[h]', min_value='1970-01-01', max_value='1970-01-11')

    data = np.array(['NaT', '1970-01-05', 'NaT', 'NaT', 'NaT', 'NaT', 'NaT', 'NaT'], dtype='datetime64[h]')
    new_data = dtype1.loads(dtype1.dumps(data), (8,))

    assert new_data[1] == data[1]
    assert np.isnat(new_data[[0, 2, 3, 4, 5, 6, 7]]).all()


def test_encode_float32_uint32_top_edge_never_casts_undefined():
    # values whose float32-scaled representation lands in the top ~256-ULP band of uint32
    # must map to fillvalue (NaN), not reach the undefined cast via float32 bound promotion
    dtype1 = dtypes.dtype('float32', precision=4, min_value=0, max_value=100000)  # uint32
    top = np.float32(429496.7295)  # scales to ~2**32 in float32
    data = np.array([50.0, top, 429496.73, 5e9], dtype='float32')
    new_data = dtype1.loads(dtype1.dumps(data), (4,))
    assert np.isclose(new_data[0], 50.0)
    assert np.isnan(new_data[1:]).all()  # unencodable top band + beyond -> NaN, never wrapped


def test_encode_packed_integer_beyond_width_raises():
    # packed ints have no reserved missing value: values whose scaled representation exceeds
    # the encoded integer width must fail loud, never wrap modulo the width (pre-fix:
    # 70000 on a 0..1000 uint16 packing round-tripped as 4464). Values inside the width but
    # outside the declared min/max remain the caller's policy (like floats).
    dtype1 = dtypes.dtype('int32', min_value=0, max_value=1000)
    ok = dtype1.loads(dtype1.dumps(np.array([0, 500, 1000], dtype='int32')), (3,))
    assert list(ok) == [0, 500, 1000]
    with pytest.raises(ValueError):
        dtype1.dumps(np.array([70000], dtype='int32'))  # scaled 70001 > uint16 max
    with pytest.raises(ValueError):
        dtype1.dumps(np.array([-5], dtype='int32'))  # scaled -4 < 0
