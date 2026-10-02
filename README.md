# Sammlung

Lightweight, cache-friendly zero-dependency collection types for Rust — bit sets, sorted maps, and sorted sets backed by vectors.

## Types

| Type | Description | Best for |
|---|---|---|
| [`BitSet`](https://docs.rs/sammlung/latest/sammlung/bitset/struct.BitSet.html) | Bit-set backed by integer array | Lower values (typically < 50k; dense) |
| [`VecSet`](https://docs.rs/sammlung/latest/sammlung/vecset/struct.VecSet.html) | Set backed by a sorted `Vec` with binary search | Small-to-medium ordered sets; fast iteration; access as slice |
| [`VecMap`](https://docs.rs/sammlung/latest/sammlung/vecmap/struct.VecMap.html) | Map backed by two parallel sorted `Vec`s with binary search | Small-to-medium key-value maps; fast iteration |


## Quick start

Add to your `Cargo.toml`:

```toml
[dependencies]
sammlung = "0.1"
```

```rust
use sammlung::{BitSet, VecMap, VecSet};

// BitSet — O(1) insert/contains with 1 bit per value
let mut set = BitSet::with_capacity(1_000_000);
set.insert(42);
assert!(set.contains(42));

// VecMap — sorted key-value pairs with binary search
let mut map = VecMap::new();
map.insert("banana", 3);
map.insert("apple", 1);
assert_eq!(map.get(&"apple"), Some(&1));

// VecSet — sorted elements with set operations
let a: VecSet<i32> = vec![1, 2, 3].into();
let b: VecSet<i32> = vec![2, 3, 4].into();
let union: Vec<_> = a.union(&b).copied().collect();
assert_eq!(union, vec![1, 2, 3, 4]);
```

## Features

(enabled by default)
- **`std`** - uses `std`

(disabled by default)
- **`unstable`** — enables unstable APIs (e.g. `Extend::extend_one`, custom allocator support on `VecSet`).

## Design philosophy

- Tuned for a **small number of small items** (typically < 2^16) where cache locality matters more than asymptotic complexity
- Favor **reuse of allocations**: Once capacity covers the values/elements you need, inserts are zero-allocation.
- **Zero dependencies** — no external crates needed.
- `no_std` compatible (with `alloc`)

For large-scale data or frequent inserts/removes, prefer the collections from the standard library.
Do measurements if `sammlung` is a better fit for your use case.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.
