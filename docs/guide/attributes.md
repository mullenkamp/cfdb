# Attributes

Attributes store metadata on datasets and individual variables. All attribute values must be JSON-serializable.

## Dataset Attributes

```python
with cfdb.open_dataset(file_path, flag='w') as ds:
    # Set
    ds.attrs['title'] = 'Example dataset'
    ds.attrs['institution'] = 'My University'

    # Get
    print(ds.attrs['title'])

    # Iterate
    for key, value in ds.attrs.items():
        print(key, value)
```

## Variable Attributes

```python
with cfdb.open_dataset(file_path, flag='w') as ds:
    temp = ds['temperature']
    temp.attrs['units'] = 'degC'
    temp.attrs['long_name'] = 'Air Temperature'
    temp.attrs['valid_range'] = [-80, 60]
```

## Dict-Like Interface

The `Attributes` class supports standard dict operations:

| Operation | Example |
|-----------|---------|
| Set | `attrs['key'] = value` or `attrs.set('key', value)` |
| Get | `attrs['key']` or `attrs.get('key')` |
| Delete | `del attrs['key']` |
| Contains | `'key' in attrs` |
| Keys | `attrs.keys()` |
| Values | `attrs.values()` |
| Items | `attrs.items()` |
| Pop | `attrs.pop('key')` |
| Update | `attrs.update({'key': 'val'})` |
| Clear | `attrs.clear()` |
| All data | `attrs.data` (returns a deep copy) |

## JSON Requirement

All values must be JSON-serializable. This includes strings, numbers, booleans, lists, and dicts (with string keys). Numpy arrays and other complex objects are not allowed:

```python
# OK
ds.attrs['scale'] = 0.1
ds.attrs['tags'] = ['climate', 'surface']
ds.attrs['config'] = {'version': 2, 'enabled': True}

# Raises ValueError
ds.attrs['data'] = np.array([1, 2, 3])  # not JSON-serializable
```

## Read-Only Access

Attributes on a read-only dataset cannot be modified:

```python
with cfdb.open_dataset(file_path, flag='r') as ds:
    print(ds.attrs['title'])      # OK
    ds.attrs['title'] = 'new'     # Raises ValueError
```

## Template Attributes

When using coordinate or data variable templates (e.g., `ds.create.coord.lat()`), standard CF attributes like `long_name`, `standard_name`, and `units` are set automatically from [cfdb-vars](https://github.com/mullenkamp/cfdb-vars).
