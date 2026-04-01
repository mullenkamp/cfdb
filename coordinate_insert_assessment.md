# Coordinate Insert Method — Feasibility Assessment

## Goal

Add an `insert()` method to the `Coordinate` class so that gaps in coordinate values (e.g., missing days in a time series) can be filled, alongside writing new data at those positions.

## Current Design

- `append()` adds values after the max — cheap, because new chunk keys are created at the end of the absolute position space. Existing chunks are untouched.
- `prepend()` adds values before the min — cheap, because it shifts the `origin` metadata field. Existing chunk keys stay at the same absolute positions.

Both operations avoid modifying any existing data variable chunks.

## Why Insert Is Expensive

Booklet stores data in chunks keyed by absolute position (e.g., `temp!40.0`). These keys **cannot be renamed** — the only way to move data to a new key is to read the value, delete the old key, and write to a new key.

The `origin` metadata trick used by `prepend` shifts the entire coordinate space uniformly, but it cannot shift "everything after position P" while leaving earlier positions alone.

Inserting N values at coordinate index P means every data variable chunk referencing positions >= P along that dimension must be:

1. Read from its current key
2. Deleted at the old key
3. Rewritten at a new key shifted forward by N

This is **O(data_after_insertion_point)** per affected data variable. Inserting near the end of a coordinate is relatively cheap; inserting near the beginning is essentially rewriting the entire dataset along that dimension.

## What the Implementation Would Look Like

### Coordinate chunks
Straightforward — merge new values into the sorted array and rewrite all coordinate chunks.

### Data variable chunks (the hard part)
A 4-phase algorithm, ordered carefully because `DataVariable.shape` is derived from coordinate metadata:

1. **Read**: For each data variable referencing this coordinate, read all data from the insertion index onward into memory.
2. **Delete**: Delete all chunk keys covering that tail region from Booklet.
3. **Update coordinate**: Rewrite coordinate chunks and update the coordinate's `shape` metadata.
4. **Write back**: Write the saved tail data at new positions (shifted by N) using `dv.set()`. This works because Phase 3 updated the shape.

### Additional constraints
- **Single-gap per call**: Multi-gap inserts (filling several disjoint gaps at once) would require non-uniform shifts, making the chunk math significantly more complex. Would need to be done as multiple `insert()` calls, back to front.
- **Memory**: All data after the insertion point must be held in memory per affected data variable during the operation.
- **Numeric/datetime only**: Geometry and string coordinates don't have a meaningful "gap" concept.

## Summary

The operation is correct and implementable, but inherently expensive due to Booklet's key-value design — there is no key rename, so shifting chunks requires full read-delete-write cycles. The cost scales with the amount of data after the insertion point, which for early-position inserts on large datasets could mean rewriting most of the stored data.
