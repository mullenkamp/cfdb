"""Exact ebooklet fetch plan (ranged GETs per group, bytes needed vs downloaded) for a selection,
computed from the remote index only — mirrors ebooklet.main.load_items + utils.get_remote_group_values."""
import numpy as np
from ebooklet import utils as eu
from cfdb import indexers

def plan(ds, var, sel):
    dv = ds[var]; ri = ds._blt._remote_index; ng = ds._blt._num_groups
    slices = indexers.index_combo_all(sel, tuple(0 for _ in dv.shape), dv.shape)
    groups = {}
    for key in indexers.slices_to_keys(slices, var, dv.chunk_shape):
        v = ri.get(key)
        if v is None:
            continue
        off, ln = eu.bytes_to_int(v[7:11]), eu.bytes_to_int(v[11:15])
        groups.setdefault(eu.key_to_group_id(key, ng), []).append((key, off, ln))
    need = fetched = 0; n = 0
    for infos in groups.values():
        infos.sort(key=lambda x: x[1])
        k0, o0, _ = infos[0]; k1, o1, l1 = infos[-1]
        start = o0 - eu.group_entry_fixed_overhead - len(k0.encode())
        fetched += o1 + l1 - start
        need += sum(l for _, _, l in infos); n += len(infos)
    return dict(chunks=n, gets=len(groups), need=need, fetched=fetched)
