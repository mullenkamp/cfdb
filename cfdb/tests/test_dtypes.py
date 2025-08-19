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

















