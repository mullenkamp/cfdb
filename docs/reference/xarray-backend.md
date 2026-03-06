# Xarray Backend

::: cfdb.xarray_backend.CfdbBackendEntrypoint
    options:
        show_bases: true
        members:
            - open_dataset
            - guess_can_open

::: cfdb.xarray_backend.CfdbDataStore
    options:
        show_bases: false
        members:
            - get_variables
            - get_attrs
            - close

::: cfdb.xarray_backend.CfdbBackendArray
    options:
        show_bases: true
