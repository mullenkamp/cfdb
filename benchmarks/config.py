DATA_TIERS = {
    'small': {
        'lat_size': 100,
        'lon_size': 100,
        'time_size': 49,
        'chunk_shape': (20, 20, 49),
        'append_time_size': 7,
        'n_iterations': 5,
    },
    'medium': {
        'lat_size': 500,
        'lon_size': 500,
        'time_size': 98,
        'chunk_shape': (50, 50, 14),
        'append_time_size': 14,
        'n_iterations': 3,
    },
    'large': {
        'lat_size': 1000,
        'lon_size': 1000,
        'time_size': 364,
        'chunk_shape': (100, 100, 28),
        'append_time_size': 28,
        'n_iterations': 1,
    },
}
