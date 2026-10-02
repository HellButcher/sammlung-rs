//! A map backed by two parallel sorted `Vec`s (keys and values) that uses binary search for lookups.
//!
//! Keys are kept in sorted order at all times, with values stored in the same order as their
//! corresponding keys. Insert and remove are O(n) due to vector shifts; get and contains are
//! O(log n).
//!
//! # Best for
//!
//! Small-to-medium maps where cache locality matters more than asymptotic lookup speed. For large
//! maps, consider `std::collections::BTreeMap` or `std::collections::HashMap` instead.
//!
//! # Entry API
//!
//! Use [`entry()`](VecMap::entry) to inspect and conditionally insert values:
//!
//! ```
//! use sammlung::VecMap;
//!
//! let mut map = VecMap::new();
//!
//! // Insert only if absent
//! map.entry("key").or_insert(42);
//!
//! // Compute value lazily
//! map.entry("key").or_insert_with(|| 99);
//! ```
//!
//! # Example
//!
//! ```
//! use sammlung::VecMap;
//!
//! let mut map = VecMap::new();
//! map.insert("banana", 3);
//! map.insert("apple", 1);
//! map.insert("cherry", 2);
//!
//! assert_eq!(map.get(&"apple"), Some(&1));
//! assert_eq!(map.len(), 3);
//! assert!(map.contains_key(&"banana"));
//! ```

use core::{fmt, ops};

use alloc::vec::Vec;

/// A map backed by two parallel sorted `Vec`s (keys and values) that uses binary search for lookups.
///
/// Keys are kept in sorted order at all times, with values stored in the same order as their
/// corresponding keys. Insert and remove are O(n) due to vector shifts; get and contains are
/// O(log n).
///
/// # Best for
///
/// Small-to-medium maps where cache locality matters more than asymptotic lookup speed. For large
/// maps, consider `BTreeMap` or `HashMap` instead.
///
/// # Example
///
/// ```
/// # use sammlung::VecMap;
///
/// let mut map = VecMap::new();
/// map.insert("banana", 3);
/// map.insert("apple", 1);
/// map.insert("cherry", 2);
///
/// assert_eq!(map.get(&"apple"), Some(&1));
/// assert_eq!(map.len(), 3);
/// assert!(map.contains_key(&"banana"));
/// ```
///
/// # Entry API
///
/// Use [`entry()`](VecMap::entry) to inspect and conditionally insert values:
///
/// ```
/// # use sammlung::VecMap;
///
/// let mut map = VecMap::new();
///
/// // Insert only if absent
/// map.entry("key").or_insert(42);
///
/// // Compute value lazily
/// map.entry("key").or_insert_with(|| 99);
/// ```
#[derive(Clone, PartialEq, Eq, Ord, PartialOrd, Hash)]
pub struct VecMap<K: Ord, V> {
    keys: Vec<K>,
    values: Vec<V>,
}

impl<K: Ord, V> VecMap<K, V> {
    /// Create a new empty `VecMap`.
    ///
    /// This is a `const fn` and can be called in const contexts.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::VecMap;
    ///
    /// let map: VecMap<i32, &str> = VecMap::new();
    /// assert!(map.is_empty());
    /// ```
    #[inline]
    pub const fn new() -> Self {
        Self {
            keys: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Create a new empty `VecMap` with reserved capacity for at least `cap` key-value pairs.
    ///
    /// # Arguments
    ///
    /// * `cap` — The minimum number of entries to reserve space for.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::VecMap;
    ///
    /// let map: VecMap<i32, i32> = VecMap::with_capacity(100);
    /// assert!(map.is_empty());
    /// ```
    #[inline]
    pub fn with_capacity(cap: usize) -> Self {
        VecMap {
            keys: Vec::with_capacity(cap),
            values: Vec::with_capacity(cap),
        }
    }

    /// Returns the number of key-value pairs the map can hold without reallocating.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.keys.capacity()
    }

    /// Reserves capacity for at least `additional` more key-value pairs.
    ///
    /// The collection may reserve more space to speculatively avoid frequent reallocations.
    /// After calling reserve, capacity will be greater than or equal to self.len() + additional. Does nothing if capacity is already sufficient.
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.keys.reserve(additional);
        self.values.reserve(additional);
    }

    /// Shrinks the capacity of the internal vectors as much as possible.
    /// The behavior of this method depends on the allocator, which may either shrink the
    /// vector in-place or reallocate. The resulting vector might still have some excess capacity,
    /// just as is the case for with_capacity.
    #[inline]
    pub fn shrink_to_fit(&mut self) {
        self.keys.shrink_to_fit();
        self.values.shrink_to_fit();
    }

    /// Insert a `(key, value)` pair in sorted order.
    ///
    /// If the key already exists, this is a no-op — the existing value is **not** overwritten.
    /// To replace an existing value, use [`entry()`](VecMap::entry) or
    /// [`OccupiedEntry::insert()`](OccupiedEntry::insert).
    ///
    /// This operation is O(n) due to the vector shift required to maintain sorted order.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::VecMap;
    ///
    /// let mut map = VecMap::new();
    /// map.insert(1, "one");
    /// map.insert(1, "uno");  // no-op: key 1 already exists
    /// assert_eq!(map.get(&1), Some(&"one"));
    /// ```
    #[inline]
    pub fn insert(&mut self, key: K, value: V) {
        match self.keys.binary_search(&key) {
            Ok(_) => {} // already present, do nothing
            Err(pos) => {
                self.keys.insert(pos, key);
                self.values.insert(pos, value);
            }
        }
    }

    /// Remove `key` and return its value if present.
    ///
    /// This operation is O(n) due to the vector shift.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::VecMap;
    ///
    /// let mut map = VecMap::new();
    /// map.insert(1, "one");
    /// assert_eq!(map.remove(&1), Some("one"));
    /// assert_eq!(map.remove(&1), None);
    /// ```
    #[inline]
    pub fn remove(&mut self, key: &K) -> Option<V> {
        let (_, value) = self.remove_entry(key)?;
        Some(value)
    }

    /// Remove `key` and return the `(key, value)` pair if present.
    ///
    /// This operation is O(n) due to the vector shift.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::VecMap;
    ///
    /// let mut map = VecMap::new();
    /// map.insert(1, "one");
    /// assert_eq!(map.remove_entry(&1), Some((1, "one")));
    /// ```
    #[inline]
    pub fn remove_entry(&mut self, key: &K) -> Option<(K, V)> {
        match self.keys.binary_search(key) {
            Ok(pos) => {
                let key = self.keys.remove(pos);
                let value = self.values.remove(pos);
                Some((key, value))
            }
            Err(_) => None,
        }
    }

    /// Get a reference to the value associated with `key`.
    ///
    /// This operation is O(log n) using binary search.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::VecMap;
    ///
    /// let mut map = VecMap::new();
    /// map.insert("x", 10);
    /// assert_eq!(map.get(&"x"), Some(&10));
    /// assert_eq!(map.get(&"y"), None);
    /// ```
    #[inline]
    pub fn get(&self, key: &K) -> Option<&V> {
        self.keys
            .binary_search(key)
            .ok()
            .map(|pos| &self.values[pos])
    }

    /// Get a mutable reference to the value associated with `key`.
    ///
    /// This operation is O(log n) using binary search.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::VecMap;
    ///
    /// let mut map = VecMap::new();
    /// map.insert(1, vec![1]);
    /// map.get_mut(&1).unwrap().push(2);
    /// assert_eq!(map.get(&1), Some(&vec![1, 2]));
    /// ```
    #[inline]
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        self.keys
            .binary_search(key)
            .ok()
            .map(move |pos| &mut self.values[pos])
    }

    /// Get references to both the key and value associated with `key`.
    ///
    /// This operation is O(log n) using binary search.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::VecMap;
    ///
    /// let mut map = VecMap::new();
    /// map.insert("answer", 42);
    /// assert_eq!(map.get_key_value(&"answer"), Some((&"answer", &42)));
    /// ```
    #[inline]
    pub fn get_key_value(&self, key: &K) -> Option<(&K, &V)> {
        self.keys
            .binary_search(key)
            .ok()
            .map(|pos| (&self.keys[pos], &self.values[pos]))
    }

    /// Get an entry for `key`, suitable for insertion into the map.
    ///
    /// Returns an [`Entry`] which is either [`Entry::Occupied`] (if the key exists) or
    /// [`Entry::Vacant`] (if it does not). The entry API enables ergonomic look-up-and-insert
    /// patterns:
    ///
    /// - [`or_insert()`](Entry::or_insert) — insert a default value if absent.
    /// - [`or_insert_with()`](Entry::or_insert_with) — lazily compute a value if absent.
    /// - [`or_insert_with_key()`](Entry::or_insert_with_key) — compute a value from the key if absent.
    /// - [`or_default()`](Entry::or_default) — use `Default::default()` if absent.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::VecMap;
    ///
    /// let mut map = VecMap::new();
    ///
    /// // Count occurrences
    /// *map.entry("hello").or_insert(0) += 1;
    /// *map.entry("hello").or_insert(0) += 1;
    /// assert_eq!(map.get(&"hello"), Some(&2));
    /// ```
    #[inline]
    pub fn entry(&mut self, key: K) -> Entry<'_, K, V> {
        match self.keys.binary_search(&key) {
            Ok(pos) => Entry::Occupied(OccupiedEntry {
                map: self,
                index: pos,
            }),
            Err(pos) => Entry::Vacant(VacantEntry {
                map: self,
                index: pos,
                key,
            }),
        }
    }

    /// Get a reference to the smallest `(key, value)` pair, or `None` if the map is empty.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::VecMap;
    ///
    /// let mut map = VecMap::new();
    /// assert!(map.first_key_value().is_none());
    ///
    /// map.insert(3, "c");
    /// map.insert(1, "a");
    /// assert_eq!(map.first_key_value(), Some((&1, &"a")));
    /// ```
    #[inline]
    pub fn first_key_value(&self) -> Option<(&K, &V)> {
        self.keys.first().zip(self.values.first())
    }

    /// Get a reference to the largest `(key, value)` pair, or `None` if the map is empty.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::VecMap;
    ///
    /// let mut map = VecMap::new();
    /// map.insert(1, "a");
    /// map.insert(3, "c");
    /// assert_eq!(map.last_key_value(), Some((&3, &"c")));
    /// ```
    #[inline]
    pub fn last_key_value(&self) -> Option<(&K, &V)> {
        self.keys.last().zip(self.values.last())
    }

    /// Get a mutable handle to the smallest entry, or `None` if the map is empty.
    ///
    /// The returned [`OccupiedEntry`] allows inspecting and mutating the value in place.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::VecMap;
    ///
    /// let mut map = VecMap::new();
    /// map.insert(1, vec![1]);
    ///
    /// if let Some(mut entry) = map.first_entry() {
    ///     entry.get_mut().push(2);
    /// }
    /// assert_eq!(map.get(&1), Some(&vec![1, 2]));
    /// ```
    #[inline]
    pub fn first_entry(&mut self) -> Option<OccupiedEntry<'_, K, V>> {
        if self.keys.is_empty() {
            None
        } else {
            Some(OccupiedEntry {
                map: self,
                index: 0,
            })
        }
    }

    /// Get a mutable handle to the largest entry, or `None` if the map is empty.
    ///
    /// The returned [`OccupiedEntry`] allows inspecting and mutating the value in place.
    #[inline]
    pub fn last_entry(&mut self) -> Option<OccupiedEntry<'_, K, V>> {
        if self.keys.is_empty() {
            None
        } else {
            let index = self.keys.len() - 1;
            Some(OccupiedEntry { map: self, index })
        }
    }

    /// Remove and return the smallest `(key, value)` pair, or `None` if empty.
    ///
    /// This operation is O(n) due to the vector shift.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::VecMap;
    ///
    /// let mut map = VecMap::new();
    /// map.insert(1, "a");
    /// map.insert(3, "c");
    /// assert_eq!(map.pop_first(), Some((1, "a")));
    /// assert_eq!(map.len(), 1);
    /// ```
    #[inline]
    pub fn pop_first(&mut self) -> Option<(K, V)> {
        if self.keys.is_empty() {
            None
        } else {
            Some((self.keys.remove(0), self.values.remove(0)))
        }
    }

    /// Remove and return the largest `(key, value)` pair, or `None` if empty.
    ///
    /// This operation is O(1).
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::VecMap;
    ///
    /// let mut map = VecMap::new();
    /// map.insert(1, "a");
    /// map.insert(3, "c");
    /// assert_eq!(map.pop_last(), Some((3, "c")));
    /// assert_eq!(map.len(), 1);
    /// ```
    #[inline]
    pub fn pop_last(&mut self) -> Option<(K, V)> {
        self.keys.pop().zip(self.values.pop())
    }

    /// Check if `key` is present in the map.
    ///
    /// This operation is O(log n) using binary search.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::VecMap;
    ///
    /// let mut map = VecMap::new();
    /// map.insert(1, "one");
    /// assert!(map.contains_key(&1));
    /// assert!(!map.contains_key(&2));
    /// ```
    #[inline]
    pub fn contains_key(&self, key: &K) -> bool {
        self.keys.binary_search(key).is_ok()
    }

    /// Returns the number of key-value pairs in the map.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::VecMap;
    ///
    /// let mut map = VecMap::new();
    /// assert_eq!(map.len(), 0);
    ///
    /// map.insert(1, "a");
    /// map.insert(2, "b");
    /// assert_eq!(map.len(), 2);
    /// ```
    #[inline]
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Returns `true` if the map contains no key-value pairs.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::VecMap;
    ///
    /// let mut map = VecMap::new();
    /// assert!(map.is_empty());
    ///
    /// map.insert(1, "a");
    /// assert!(!map.is_empty());
    /// ```
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Remove all key-value pairs from the map, keeping the allocations.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::VecMap;
    ///
    /// let mut map = VecMap::new();
    /// map.insert(1, "a");
    /// map.clear();
    /// assert!(map.is_empty());
    /// ```
    #[inline]
    pub fn clear(&mut self) {
        self.keys.clear();
        self.values.clear();
    }

    /// Consumes the map, returning the keys vector.
    ///
    /// The returned `Vec<K>` is sorted and contains no duplicates.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::VecMap;
    ///
    /// let mut map = VecMap::new();
    /// map.insert(2, "b");
    /// map.insert(1, "a");
    ///
    /// let keys = map.into_keys();
    /// assert_eq!(keys, vec![1, 2]);
    /// ```
    #[inline]
    pub fn into_keys(self) -> Vec<K> {
        self.keys
    }

    /// Consumes the map, returning the values vector.
    ///
    /// The returned `Vec<V>` is in the same order as the corresponding keys.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::VecMap;
    ///
    /// let mut map = VecMap::new();
    /// map.insert(2, "b");
    /// map.insert(1, "a");
    ///
    /// let values = map.into_values();
    /// assert_eq!(values, vec!["a", "b"]);
    /// ```
    #[inline]
    pub fn into_values(self) -> Vec<V> {
        self.values
    }

    /// Returns a slice of the keys in sorted order.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::VecMap;
    ///
    /// let mut map = VecMap::new();
    /// map.insert(2, "b");
    /// map.insert(1, "a");
    ///
    /// assert_eq!(map.keys(), &[1, 2]);
    /// ```
    #[inline]
    pub fn keys(&self) -> &[K] {
        &self.keys
    }

    /// Returns a slice of the values in key-sorted order.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::VecMap;
    ///
    /// let mut map = VecMap::new();
    /// map.insert(2, "b");
    /// map.insert(1, "a");
    ///
    /// assert_eq!(map.values(), &["a", "b"]);
    /// ```
    #[inline]
    pub fn values(&self) -> &[V] {
        &self.values
    }

    /// Returns a mutable slice of the values in key-sorted order.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::VecMap;
    ///
    /// let mut map = VecMap::new();
    /// map.insert(1, 0);
    /// map.values_mut()[0] = 42;
    /// assert_eq!(map.get(&1), Some(&42));
    /// ```
    #[inline]
    pub fn values_mut(&mut self) -> &mut [V] {
        &mut self.values
    }

    /// Returns an iterator over `(key, value)` pairs in sorted key order.
    ///
    /// Yields `(&'a K, &'a V)` shared references.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::VecMap;
    ///
    /// let mut map = VecMap::new();
    /// map.insert(2, "b");
    /// map.insert(1, "a");
    ///
    /// let pairs: Vec<_> = map.iter().collect();
    /// assert_eq!(pairs, vec![(&1, &"a"), (&2, &"b")]);
    /// ```
    #[inline]
    pub fn iter(&self) -> Iter<'_, K, V> {
        self.keys.iter().zip(self.values.iter())
    }

    /// Returns an iterator over `(key, mutable value)` pairs in sorted key order.
    ///
    /// Yields `(&'a K, &'a mut V)` pairs.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::VecMap;
    ///
    /// let mut map = VecMap::new();
    /// map.insert(1, vec![1]);
    /// for (_, v) in map.iter_mut() {
    ///     v.push(2);
    /// }
    /// assert_eq!(map.get(&1), Some(&vec![1, 2]));
    /// ```
    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<'_, K, V> {
        self.keys.iter().zip(self.values.iter_mut())
    }

    /// Retains only the entries for which the predicate returns `true`.
    ///
    /// Elements for which `f(key, value)` returns `false` are removed. The predicate receives a
    /// mutable reference to the value, allowing mutation of retained entries.
    ///
    /// This operation is O(n²) in the worst case (all elements removed), as each removal shifts
    /// the vectors.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::VecMap;
    ///
    /// let mut map = VecMap::new();
    /// map.insert(1, 10);
    /// map.insert(2, 20);
    /// map.insert(3, 30);
    ///
    /// map.retain(|&key, _| key > 1);
    /// assert_eq!(map.len(), 2); // keys {2, 3}
    /// ```
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&K, &mut V) -> bool,
    {
        let mut i = 0;
        while i < self.keys.len() {
            if !f(&self.keys[i], &mut self.values[i]) {
                self.keys.remove(i);
                self.values.remove(i);
            } else {
                i += 1;
            }
        }
    }
}

impl<K: Ord, V> Default for VecMap<K, V> {
    /// Creates an empty `VecMap`.
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Ord, V> ops::Index<&K> for VecMap<K, V> {
    type Output = V;

    /// Index into the map by key. Panics if the key is not found.
    ///
    /// # Panics
    ///
    /// Panics if `key` is not present in the map. Use [`get()`](VecMap::get) for a non-panicking
    /// alternative.
    ///
    /// # Example
    ///
    /// ```should_panic
    /// # use sammlung::VecMap;
    ///
    /// let mut map = VecMap::new();
    /// map.insert(1, "one");
    /// println!("{}", map[&1]);  // prints "one"
    /// println!("{}", map[&2]);  // panics!
    /// ```
    fn index(&self, key: &K) -> &Self::Output {
        self.get(key).expect("key not found")
    }
}

impl<K: Ord, V> ops::IndexMut<&K> for VecMap<K, V> {
    /// Mutable index into the map by key. Panics if the key is not found.
    ///
    /// # Panics
    ///
    /// Panics if `key` is not present in the map.
    fn index_mut(&mut self, key: &K) -> &mut Self::Output {
        self.get_mut(key).expect("key not found")
    }
}

impl<K: Ord, V> Extend<(K, V)> for VecMap<K, V> {
    /// Extends the map with key-value pairs from an iterator.
    ///
    /// Each pair is inserted in sorted order. If a key already exists, it is not overwritten.
    #[inline]
    fn extend<T: IntoIterator<Item = (K, V)>>(&mut self, iter: T) {
        let iter = iter.into_iter();
        let (size_hint, _) = iter.size_hint();
        if self.keys.capacity() < size_hint {
            self.keys.reserve(size_hint - self.keys.len());
            self.values.reserve(size_hint - self.values.len());
        }
        for (key, value) in iter {
            self.insert(key, value);
        }
    }
}

impl<K: Ord, V> FromIterator<(K, V)> for VecMap<K, V> {
    /// Collects key-value pairs from an iterator into a `VecMap`.
    #[inline]
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
        let iter = iter.into_iter();
        let (size_hint, _) = iter.size_hint();
        let mut map = VecMap::with_capacity(size_hint);
        map.extend(iter);
        map
    }
}

impl<K: Ord, V, const N: usize> From<[(K, V); N]> for VecMap<K, V> {
    /// Converts an array of `(K, V)` pairs into a `VecMap`.
    ///
    /// Later duplicate keys silently overwrite earlier ones (via [`insert()`](VecMap::insert)).
    #[inline]
    fn from(array: [(K, V); N]) -> Self {
        let mut map = VecMap::with_capacity(N);
        map.extend(array);
        map
    }
}

pub type IntoIter<K, V> = core::iter::Zip<alloc::vec::IntoIter<K>, alloc::vec::IntoIter<V>>;
pub type Iter<'a, K, V> = core::iter::Zip<core::slice::Iter<'a, K>, core::slice::Iter<'a, V>>;
pub type IterMut<'a, K, V> = core::iter::Zip<core::slice::Iter<'a, K>, core::slice::IterMut<'a, V>>;

impl<K: Ord, V> IntoIterator for VecMap<K, V> {
    type Item = (K, V);
    type IntoIter = IntoIter<K, V>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.keys.into_iter().zip(self.values)
    }
}

impl<'a, K: Ord, V> IntoIterator for &'a VecMap<K, V> {
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, K: Ord, V> IntoIterator for &'a mut VecMap<K, V> {
    type Item = (&'a K, &'a mut V);
    type IntoIter = IterMut<'a, K, V>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

/// The entry API for looking up a key in a [`VecMap`], which may return either an existing entry
/// or a vacant slot for insertion.
///
/// Obtain an `Entry` via [`VecMap::entry()`]. It is an enum with two variants:
///
/// - [`Entry::Occupied`] — the key exists; provides methods to inspect and mutate the value.
/// - [`Entry::Vacant`] — the key does not exist; provides methods to insert a new value.
///
/// # Example
///
/// ```
/// # use sammlung::VecMap;
/// # use sammlung::vecmap::Entry;
///
/// let mut map = VecMap::new();
///
/// match map.entry(1) {
///     Entry::Vacant(v) => {
///         v.insert("one");
///     }
///     Entry::Occupied(o) => {
///         println!("Key already exists: {}", o.get());
///     }
/// }
/// ```
pub enum Entry<'a, K: Ord, V> {
    /// An occupied entry — the key exists in the map.
    Occupied(OccupiedEntry<'a, K, V>),
    /// A vacant entry — the key does not exist; insertion is possible.
    Vacant(VacantEntry<'a, K, V>),
}

/// A handle to an existing key-value pair in a [`VecMap`].
///
/// Obtained from [`Entry::Occupied`]. Provides methods to read, mutate, and remove the entry.
///
/// # Example
///
/// ```
/// # use sammlung::VecMap;
///
/// let mut map = VecMap::new();
/// map.insert(1, vec![1]);
///
/// if let Some(mut entry) = map.first_entry() {
///     entry.get_mut().push(2);
///     println!("Value: {:?}", entry.get());  // [1, 2]
/// }
/// ```
pub struct OccupiedEntry<'a, K: Ord, V> {
    map: &'a mut VecMap<K, V>,
    index: usize,
}

/// A handle to a vacant slot in a [`VecMap`] where a new key-value pair can be inserted.
///
/// Obtained from [`Entry::Vacant`]. Provides methods to insert a value at the vacant position.
///
/// # Example
///
/// ```
/// # use sammlung::VecMap;
/// # use sammlung::vecmap::Entry;
///
/// let mut map = VecMap::new();
///
/// match map.entry(1) {
///     Entry::Vacant(v) => {
///         v.insert("one");
///     }
///     Entry::Occupied(_) => panic!("unexpected"),
/// }
/// assert_eq!(map.get(&1), Some(&"one"));
/// ```
pub struct VacantEntry<'a, K: Ord, V> {
    map: &'a mut VecMap<K, V>,
    index: usize,
    key: K,
}

impl<'a, K: Ord, V> Entry<'a, K, V> {
    /// Returns the index of this entry in the underlying key and value vectors.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::VecMap;
    ///
    /// let mut map = VecMap::new();
    /// map.insert(1, "a");
    /// map.insert(2, "b");
    ///
    /// let entry = map.entry(2);
    /// assert_eq!(entry.index(), 1);
    /// ```
    #[inline]
    pub fn index(&self) -> usize {
        match self {
            Entry::Occupied(entry) => entry.index,
            Entry::Vacant(entry) => entry.index,
        }
    }

    /// Returns a reference to the key of this entry.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::VecMap;
    ///
    /// let mut map = VecMap::new();
    /// map.insert(42, "answer");
    ///
    /// let entry = map.entry(42);
    /// assert_eq!(entry.key(), &42);
    /// ```
    #[inline]
    pub fn key(&self) -> &K {
        match self {
            Entry::Occupied(entry) => entry.key(),
            Entry::Vacant(entry) => entry.key(),
        }
    }

    /// Ensures a value is in the entry by inserting the default if vacant, or returning a mutable
    /// reference to the existing value.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::VecMap;
    ///
    /// let mut map = VecMap::new();
    ///
    /// *map.entry(1).or_insert(0) += 1;
    /// *map.entry(1).or_insert(0) += 1;
    /// assert_eq!(map.get(&1), Some(&2));
    /// ```
    #[inline]
    pub fn or_insert(self, value: V) -> &'a mut V {
        match self {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(value),
        }
    }

    /// Ensures a value is in the entry by inserting the result of the closure if vacant, or
    /// returning a mutable reference to the existing value.
    ///
    /// The closure is only called when the entry is vacant.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::VecMap;
    ///
    /// let mut map = VecMap::new();
    ///
    /// // Expensive computation only happens on first insert
    /// let val = map.entry(1).or_insert_with(|| {
    ///     println!("Computing...");
    ///     42
    /// });
    /// assert_eq!(*val, 42);
    /// ```
    #[inline]
    pub fn or_insert_with(self, f: impl FnOnce() -> V) -> &'a mut V {
        match self {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(f()),
        }
    }

    /// Ensures a value is in the entry by inserting the result of the closure (which receives
    /// the key) if vacant, or returning a mutable reference to the existing value.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::VecMap;
    ///
    /// let mut map = VecMap::new();
    ///
    /// let val = map.entry("hello").or_insert_with_key(|k| {
    ///     format!("greeting: {}", k)
    /// });
    /// assert_eq!(val, "greeting: hello");
    /// ```
    #[inline]
    pub fn or_insert_with_key(self, f: impl FnOnce(&K) -> V) -> &'a mut V {
        match self {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                let value = f(entry.key());
                entry.insert(value)
            }
        }
    }

    /// Ensures a value is in the entry by inserting `Default::default()` if vacant, or returning
    /// a mutable reference to the existing value.
    ///
    /// Requires `V: Default`.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::VecMap;
    ///
    /// let mut map = VecMap::new();
    ///
    /// *map.entry(1).or_default() += 1;
    /// assert_eq!(map.get(&1), Some(&1));
    /// ```
    #[inline]
    pub fn or_default(self) -> &'a mut V
    where
        V: Default,
    {
        self.or_insert_with(Default::default)
    }
}

impl<'a, K: Ord, V> OccupiedEntry<'a, K, V> {
    /// Returns the index of this entry in the underlying key and value vectors.
    #[inline]
    pub fn index(&self) -> usize {
        self.index
    }

    /// Returns a reference to the key of this entry.
    #[inline]
    pub fn key(&self) -> &K {
        &self.map.keys[self.index]
    }

    /// Consumes the entry, returning a long-lived reference to the key.
    ///
    /// The returned reference is valid for the lifetime `'a` (the same lifetime as the parent
    /// `VecMap`'s mutable borrow).
    #[inline]
    pub fn into_key(self) -> &'a K {
        &self.map.keys[self.index]
    }

    /// Returns a shared reference to the value of this entry.
    #[inline]
    pub fn get(&self) -> &V {
        &self.map.values[self.index]
    }

    /// Returns a mutable reference to the value of this entry.
    #[inline]
    pub fn get_mut(&mut self) -> &mut V {
        &mut self.map.values[self.index]
    }

    /// Consumes the entry, returning a long-lived mutable reference to the value.
    ///
    /// The returned reference is valid for the lifetime `'a` (the same lifetime as the parent
    /// `VecMap`'s mutable borrow).
    #[inline]
    pub fn into_mut(self) -> &'a mut V {
        &mut self.map.values[self.index]
    }

    /// Removes and returns both the key and value of this entry.
    ///
    /// This operation is O(n) due to the vector shift.
    #[inline]
    pub fn remove_entry(self) -> (K, V) {
        let key = self.map.keys.remove(self.index);
        let value = self.map.values.remove(self.index);
        (key, value)
    }

    /// Removes and returns the value of this entry.
    ///
    /// This operation is O(n) due to the vector shift.
    #[inline]
    pub fn remove(self) -> V {
        self.remove_entry().1
    }

    /// Replaces the value with a new one, returning the old value.
    ///
    /// Unlike [`VecMap::insert()`], this **does** overwrite an existing value.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::VecMap;
    ///
    /// let mut map = VecMap::new();
    /// map.insert(1, "one");
    ///
    /// if let Some(mut entry) = map.first_entry() {
    ///     let old = entry.insert("uno");
    ///     assert_eq!(old, "one");
    /// }
    /// assert_eq!(map.get(&1), Some(&"uno"));
    /// ```
    #[inline]
    pub fn insert(&mut self, value: V) -> V {
        core::mem::replace(self.get_mut(), value)
    }
}

impl<'a, K: Ord, V> VacantEntry<'a, K, V> {
    /// Returns the index where insertion will occur.
    #[inline]
    pub fn index(&self) -> usize {
        self.index
    }

    /// Returns a reference to the key of this vacant entry.
    #[inline]
    pub fn key(&self) -> &K {
        &self.key
    }

    /// Consumes the entry, returning the owned key.
    #[inline]
    pub fn into_key(self) -> K {
        self.key
    }

    /// Inserts a value into the map at this vacant position, returning a mutable reference to it.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::VecMap;
    /// # use sammlung::vecmap::Entry;
    ///
    /// let mut map = VecMap::new();
    ///
    /// match map.entry(1) {
    ///     Entry::Vacant(v) => {
    ///         *v.insert(42) = 99;
    ///     }
    ///     Entry::Occupied(_) => panic!(),
    /// }
    /// assert_eq!(map.get(&1), Some(&99));
    /// ```
    #[inline]
    pub fn insert(self, value: V) -> &'a mut V {
        self.insert_entry(value).into_mut()
    }

    /// Inserts a value into the map at this vacant position and returns an [`OccupiedEntry`]
    /// handle for further mutation.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::VecMap;
    /// # use sammlung::vecmap::Entry;
    ///
    /// let mut map = VecMap::new();
    ///
    /// match map.entry(1) {
    ///     Entry::Vacant(v) => {
    ///         let mut occupied = v.insert_entry(vec![1]);
    ///         occupied.get_mut().push(2);
    ///     }
    ///     _ => panic!(),
    /// }
    /// assert_eq!(map.get(&1), Some(&vec![1, 2]));
    /// ```
    #[inline]
    pub fn insert_entry(self, value: V) -> OccupiedEntry<'a, K, V> {
        self.map.keys.insert(self.index, self.key);
        self.map.values.insert(self.index, value);
        OccupiedEntry {
            map: self.map,
            index: self.index,
        }
    }
}

impl<K: Ord + fmt::Debug, V: fmt::Debug> fmt::Debug for VecMap<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map()
            .entries(self.keys.iter().zip(self.values.iter()))
            .finish()
    }
}

impl<K: Ord + fmt::Debug, V> fmt::Debug for VacantEntry<'_, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("VacantEntry").field(self.key()).finish()
    }
}

impl<K: Ord + fmt::Debug, V: fmt::Debug> fmt::Debug for OccupiedEntry<'_, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OccupiedEntry")
            .field("key", self.key())
            .field("Value", self.get())
            .finish()
    }
}

impl<K: Ord + fmt::Debug, V: fmt::Debug> fmt::Debug for Entry<'_, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Entry::Occupied(ref entry) => f.debug_tuple("Entry").field(entry).finish(),
            Entry::Vacant(ref entry) => f.debug_tuple("Entry").field(entry).finish(),
        }
    }
}
