//! Lightweight, cache and allocation friendly zero-dependency collection types for Rust — bit sets, sorted maps, and sorted sets.
//!
//! # Overview
//!
//! Sammlung (German for "collection") is tuned for a small number (< 50k) of small items where
//! cache locality matters more than asymptotic complexity:
//!
//! - [`BitSet`] — a bit-set backed by array of integers. O(1) insert, remove,and contains operations
//!   with one bit per value.
//!   Memory cost scales with the highest *value*, not the number of set bits,
//!   so it is designed for lower values (best for a dense values from 0 to highest value; and dense
//!   domains).
//!   Supports standard set operations (union, intersection, difference, symmetric difference.
//! - [`VecSet<T>`] — a set backed by a sorted `Vec` with binary search. O(log n) contains, O(n)
//!   insert/remove. Fast iteration (like slice). Dereferencable to a slice.
//! - [`VecMap<K,V>`] — a map backed by two parallel sorted `Vec`s with binary search for lookups. O(log n)
//!   get/contains, O(n) insert/remove. Fast iteration (like slice). Keys and values can be accessed
//!   seperately as slices. Includes an entry API for ergonomic look-up-and-insert patterns.
//!
//! All types favor reuse of allocations: Once capacity covers the values/elements you need, inserts are zero-allocation.
//!
//! For large-scale data or frequent inserts/removes, prefer the collections from `std` or other crates.
//! Do measurements to determine, if `sammlung` is a better fit for your use case.
//!
//! # Example
//!
//! ```
//! use sammlung::{BitSet, VecMap, VecSet};
//!
//! // BitSet — O(1) insert/contains with 1 bit per value
//! let mut set = BitSet::with_capacity(1_000);
//! set.insert(42);
//! assert!(set.contains(42));
//!
//! // VecMap — sorted key-value pairs with binary search
//! let mut map = VecMap::new();
//! map.insert("apple", 1);
//! map.insert("banana", 2);
//! assert_eq!(map.get(&"apple"), Some(&1));
//!
//! // VecSet — sorted elements with set operations
//! let a: VecSet<i32> = vec![1, 2, 3].into();
//! let b: VecSet<i32> = vec![2, 3, 4].into();
//! assert_eq!(a.union(&b).copied().collect::<Vec<_>>(), vec![1, 2, 3, 4]);
//! ```

#![no_std]
#![cfg_attr(feature = "unstable", feature(allocator_api, extend_one))]

extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

pub mod bitset;
pub mod vecmap;
pub mod vecset;

pub use bitset::BitSet;
pub use vecmap::VecMap;
pub use vecset::VecSet;
