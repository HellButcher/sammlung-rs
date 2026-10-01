use core::fmt;
use core::{
    iter::{FusedIterator, Peekable},
    ops,
};

/// A set backed by a sorted `Vec<T>` with binary search for lookups.
///
/// Insert and remove are O(n) due to vector shifts; contains is O(log n).
/// Excellent cache locality for small sets.
///
/// # Generic type parameter
///
/// `T` must implement [`Ord`] so that elements can be kept in sorted order. This enables
/// binary search for O(log n) lookups and efficient set-diff iterators that merge two sorted
/// sequences.
///
/// # Construction from unsorted data
///
/// Use [`From<Vec<T>>`](VecSet::from), [`FromIterator`], or [`extend()`](Extend::extend) to
/// construct a `VecSet` from unsorted data — these automatically sort and deduplicate.
/// For already-sorted data, use [`from_sorted_vec()`](VecSet::from_sorted_vec) or
/// [`from_sorted_iter()`](VecSet::from_sorted_iter) to avoid redundant sorting.
///
/// # Example
///
/// ```
/// # use sammlung::VecSet;
///
/// let mut set: VecSet<i32> = VecSet::new();
/// assert!(set.insert(3));
/// assert!(set.insert(1));
/// assert!(!set.insert(3));  // already present
/// assert_eq!(set.len(), 2);
/// assert!(set.contains(&3));
/// ```
///
/// # Set operations
///
/// `VecSet` supports standard set operations via both method calls and operator overloads:
///
/// | Operation | Method | Operator |
/// |---|---|---|
/// | Union | `.union(other)` | `a \| b` |
/// | Intersection | `.intersection(other)` | `a & b` |
/// | Difference | `.difference(other)` | `a - b` |
/// | Symmetric difference | `.symmetric_difference(other)` | `a ^ b` |
///
/// The iterator-based methods return zero-allocation iterators that borrow both sets. The operator
/// variants allocate and return a new `VecSet`. In-place operators (`|=`, `&=`, `^=`, `-=`)
/// mutate `self` directly.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VecSet<T: Ord> {
    values: Vec<T>,
}

impl<T: Ord> VecSet<T> {
    /// Create a new empty `VecSet`.
    ///
    /// This is a `const fn` and can be called in const contexts.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::VecSet;
    ///
    /// let set: VecSet<i32> = VecSet::new();
    /// assert!(set.is_empty());
    /// ```
    #[inline]
    pub const fn new() -> Self {
        VecSet { values: Vec::new() }
    }

    /// Create a new empty `VecSet` with reserved capacity for at least `cap` elements.
    ///
    /// # Arguments
    ///
    /// * `cap` — The minimum number of elements to reserve space for.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::VecSet;
    ///
    /// let set: VecSet<i32> = VecSet::with_capacity(100);
    /// assert!(set.is_empty());
    /// ```
    #[inline]
    pub fn with_capacity(cap: usize) -> Self {
        VecSet {
            values: Vec::with_capacity(cap),
        }
    }

    /// Returns the number of elements the set can hold without reallocating.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.values.capacity()
    }

    /// Reserve capacity for at least `additional` more elements to be inserted.
    /// The collection may reserve more space to speculatively avoid frequent reallocations.
    /// After calling reserve, capacity will be greater than or equal to self.len() + additional. Does nothing if capacity is already sufficient.
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.values.reserve(additional);
    }

    /// Shrink the capacity of the internal vector as much as possible.
    /// The behavior of this method depends on the allocator, which may either shrink the vector
    /// in-place or reallocate. The resulting vector might still have some excess capacity, just as is the case for with_capacity.
    #[inline]
    pub fn shrink_to_fit(&mut self) {
        self.values.shrink_to_fit();
    }

    /// Construct a `VecSet` from an already-sorted `Vec<T>` without deduplication or re-sorting.
    ///
    /// **Caller must guarantee** that the input is sorted in ascending order and contains no
    /// duplicates. Violating this precondition results in undefined behavior for set operations.
    ///
    /// This is a `const fn` and can be called in const contexts.
    ///
    /// # Arguments
    ///
    /// * `values` — A sorted, deduplicated vector of elements.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::VecSet;
    ///
    /// let values = vec![1, 2, 3, 5, 8];
    /// let set = VecSet::from_sorted_vec(values);
    /// assert_eq!(set.len(), 5);
    /// ```
    #[inline]
    pub const fn from_sorted_vec(values: Vec<T>) -> Self {
        VecSet { values }
    }

    /// Construct a `VecSet` from an iterator over sorted, deduplicated elements.
    ///
    /// **Caller must guarantee** that the iterator yields elements in ascending order with no
    /// duplicates. Violating this precondition results in undefined behavior for set operations.
    ///
    /// # Arguments
    ///
    /// * `iter` — An iterator yielding sorted, unique elements of type `T`.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::VecSet;
    ///
    /// let set = VecSet::from_sorted_iter(vec![1, 3, 5].into_iter());
    /// assert_eq!(set.len(), 3);
    /// ```
    #[inline]
    pub fn from_sorted_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        VecSet {
            values: iter.into_iter().collect(),
        }
    }

    /// Perform an in-place union with a sorted iterator.
    ///
    /// Inserts each element from `iter` into `self` if it is not already present, maintaining
    /// sorted order throughout. This is more efficient than calling [`insert()`](VecSet::insert)
    /// repeatedly when the iterator itself is sorted.
    ///
    /// **Note:** `iter` does not need to implement `Ord`; the method assumes the iterator yields
    /// elements in ascending order.
    ///
    /// # Arguments
    ///
    /// * `iter` — An iterator yielding elements in ascending sorted order.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::VecSet;
    ///
    /// let mut set: VecSet<i32> = VecSet::new();
    /// set.insert(1);
    /// set.insert(3);
    ///
    /// set.union_assign_sorted(vec![2, 4, 6]);
    /// assert_eq!(set.len(), 5); // {1, 2, 3, 4, 6}
    /// ```
    pub fn union_assign_sorted<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        let iter = iter.into_iter();
        let (min_size, _) = iter.size_hint();
        if self.values.capacity() < min_size {
            self.values.reserve(min_size - self.values.len());
        }
        let mut offset = 0;
        for value in iter {
            match self.values[offset..].binary_search(&value) {
                Ok(idx) => {
                    // already present
                    // update offset to start after this index
                    offset += idx + 1;
                }
                Err(idx) => {
                    // not present, insert it
                    self.values.insert(offset + idx, value);
                    // update the range to start after this index
                    offset += idx + 1;
                }
            }
        }
    }

    fn symmetric_difference_assign_sorted<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        let iter = iter.into_iter();
        let (min_size, _) = iter.size_hint();
        if self.values.capacity() < min_size {
            self.values.reserve(min_size - self.values.len());
        }
        let mut offset = 0;
        for value in iter {
            match self.values[offset..].binary_search(&value) {
                Ok(idx) => {
                    // already present, remove it
                    self.values.remove(offset + idx);
                    // update the range to start after this index
                    offset += idx;
                }
                Err(idx) => {
                    // not present, insert it
                    self.values.insert(offset + idx, value);
                    // update the range to start after this index
                    offset += idx + 1;
                }
            }
        }
    }

    /// Clear all values from the set.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::VecSet;
    ///
    /// let mut set: VecSet<i32> = vec![1, 2, 3].into();
    /// set.clear();
    /// assert!(set.is_empty());
    /// ```
    #[inline]
    pub fn clear(&mut self) {
        self.values.clear();
    }

    /// Insert `value` in sorted order.
    ///
    /// Returns `true` if the value was newly inserted (it was not already present), and `false`
    /// if the value was already in the set (no-op).
    ///
    /// This operation is O(n) due to the vector shift required to maintain sorted order.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::VecSet;
    ///
    /// let mut set: VecSet<i32> = VecSet::new();
    /// assert!(set.insert(5));   // newly inserted
    /// assert!(!set.insert(5));  // already present
    /// assert_eq!(set.len(), 1);
    /// ```
    #[inline]
    pub fn insert(&mut self, value: T) -> bool {
        match self.values.binary_search(&value) {
            Ok(_) => false, // already present
            Err(pos) => {
                self.values.insert(pos, value);
                true
            }
        }
    }

    /// Remove `value` if present in the set.
    ///
    /// Returns `true` if the value was found and removed, and `false` if it was not present.
    ///
    /// This operation is O(n) due to the vector shift required to maintain sorted order.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::VecSet;
    ///
    /// let mut set: VecSet<i32> = vec![1, 2, 3].into();
    /// assert!(set.remove(&2));
    /// assert!(!set.remove(&2)); // not present
    /// assert_eq!(set.len(), 2);
    /// ```
    #[inline]
    pub fn remove(&mut self, value: &T) -> bool {
        match self.values.binary_search(value) {
            Ok(pos) => {
                self.values.remove(pos);
                true
            }
            Err(_) => false,
        }
    }

    /// Check if `value` is present in the set.
    ///
    /// This operation is O(log n) using binary search.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::VecSet;
    ///
    /// let set: VecSet<i32> = vec![1, 2, 3].into();
    /// assert!(set.contains(&2));
    /// assert!(!set.contains(&4));
    /// ```
    #[inline]
    pub fn contains(&self, value: &T) -> bool {
        self.values.binary_search(value).is_ok()
    }

    /// Returns the number of elements in the set.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::VecSet;
    ///
    /// let set: VecSet<i32> = vec![1, 2, 3].into();
    /// assert_eq!(set.len(), 3);
    /// ```
    #[inline]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns `true` if the set contains no elements.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::VecSet;
    ///
    /// let set: VecSet<i32> = VecSet::new();
    /// assert!(set.is_empty());
    ///
    /// let mut set = set;
    /// set.insert(1);
    /// assert!(!set.is_empty());
    /// ```
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Returns an iterator over the elements in ascending order.
    ///
    /// The iterator yields `&T` references to each element.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::VecSet;
    ///
    /// let set: VecSet<i32> = vec![3, 1, 2].into();
    /// let collected: Vec<_> = set.iter().copied().collect();
    /// assert_eq!(collected, vec![1, 2, 3]); // sorted order
    /// ```
    #[inline]
    pub fn iter(&self) -> Iter<'_, T> {
        self.values.iter()
    }

    /// Returns a reference to the smallest element, or `None` if the set is empty.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::VecSet;
    ///
    /// let set: VecSet<i32> = vec![3, 1, 2].into();
    /// assert_eq!(set.first(), Some(&1));
    /// ```
    #[inline]
    pub fn first(&self) -> Option<&T> {
        self.values.first()
    }

    /// Returns a reference to the largest element, or `None` if the set is empty.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::VecSet;
    ///
    /// let set: VecSet<i32> = vec![1, 3, 2].into();
    /// assert_eq!(set.last(), Some(&3));
    /// ```
    #[inline]
    pub fn last(&self) -> Option<&T> {
        self.values.last()
    }

    /// Removes and returns the smallest element, or `None` if the set is empty.
    ///
    /// This operation is O(n) due to the vector shift.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::VecSet;
    ///
    /// let mut set: VecSet<i32> = vec![1, 2, 3].into();
    /// assert_eq!(set.pop_first(), Some(1));
    /// assert_eq!(set.pop_first(), Some(2));
    /// assert_eq!(set.len(), 1);
    /// ```
    pub fn pop_first(&mut self) -> Option<T> {
        if self.values.is_empty() {
            None
        } else {
            Some(self.values.remove(0))
        }
    }

    /// Removes and returns the largest element, or `None` if the set is empty.
    ///
    /// This operation is O(1).
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::VecSet;
    ///
    /// let mut set: VecSet<i32> = vec![1, 2, 3].into();
    /// assert_eq!(set.pop_last(), Some(3));
    /// assert_eq!(set.len(), 2);
    /// ```
    #[inline]
    pub fn pop_last(&mut self) -> Option<T> {
        self.values.pop()
    }

    /// Returns `true` if the set shares no elements with `other`.
    ///
    /// This is a linear scan over both sorted sets.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::VecSet;
    ///
    /// let a: VecSet<i32> = vec![1, 2].into();
    /// let b: VecSet<i32> = vec![3, 4].into();
    /// let c: VecSet<i32> = vec![2, 3].into();
    ///
    /// assert!(a.is_disjoint(&b));
    /// assert!(!a.is_disjoint(&c));
    /// ```
    pub fn is_disjoint(&self, other: &Self) -> bool {
        let mut i = 0;
        let mut j = 0;
        while i < self.len() && j < other.len() {
            match self.values[i].cmp(&other.values[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => return false,
            }
        }
        true
    }

    /// Returns `true` if every element of `self` is also in `other` (i.e., `self` is a subset of
    /// `other`).
    ///
    /// This is a linear scan over both sorted sets.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::VecSet;
    ///
    /// let a: VecSet<i32> = vec![1, 2].into();
    /// let b: VecSet<i32> = vec![1, 2, 3].into();
    ///
    /// assert!(a.is_subset(&b));
    /// assert!(!b.is_subset(&a));
    /// ```
    pub fn is_subset(&self, other: &Self) -> bool {
        let mut i = 0;
        let mut j = 0;
        while i < self.len() && j < other.len() {
            match self.values[i].cmp(&other.values[j]) {
                std::cmp::Ordering::Less => return false,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    i += 1;
                    j += 1;
                }
            }
        }
        i == self.len()
    }

    /// Returns `true` if every element of `other` is also in `self` (i.e., `self` is a superset
    /// of `other`).
    ///
    /// Equivalent to `other.is_subset(self)`.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::VecSet;
    ///
    /// let a: VecSet<i32> = vec![1, 2, 3].into();
    /// let b: VecSet<i32> = vec![1, 2].into();
    ///
    /// assert!(a.is_superset(&b));
    /// assert!(!b.is_superset(&a));
    /// ```
    #[inline]
    pub fn is_superset(&self, other: &Self) -> bool {
        other.is_subset(self)
    }

    /// Returns an iterator yielding elements in `self` but not in `other`.
    ///
    /// The iterator borrows both sets and does not allocate. Elements are yielded in ascending
    /// order as `&T` references.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::VecSet;
    ///
    /// let a: VecSet<i32> = vec![1, 2, 3, 4].into();
    /// let b: VecSet<i32> = vec![3, 4, 5, 6].into();
    ///
    /// let diff: Vec<_> = a.difference(&b).copied().collect();
    /// assert_eq!(diff, vec![1, 2]);
    /// ```
    #[inline]
    pub fn difference<'a>(&'a self, other: &'a Self) -> Difference<'a, T> {
        Difference {
            added: self.values.iter(),
            removed: other.values.iter().peekable(),
        }
    }

    /// Returns an iterator yielding elements in exactly one of `self` or `other` (but not both).
    ///
    /// The iterator borrows both sets and does not allocate. Elements are yielded in ascending
    /// order as `&T` references.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::VecSet;
    ///
    /// let a: VecSet<i32> = vec![1, 2, 3].into();
    /// let b: VecSet<i32> = vec![3, 4, 5].into();
    ///
    /// let sym_diff: Vec<_> = a.symmetric_difference(&b).copied().collect();
    /// assert_eq!(sym_diff, vec![1, 2, 4, 5]);
    /// ```
    #[inline]
    pub fn symmetric_difference<'a>(&'a self, other: &'a Self) -> SymmetricDifference<'a, T> {
        SymmetricDifference {
            left: self.values.iter().peekable(),
            right: other.values.iter().peekable(),
        }
    }

    /// Returns an iterator yielding elements common to both `self` and `other`.
    ///
    /// The iterator borrows both sets and does not allocate. Elements are yielded in ascending
    /// order as `&T` references.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::VecSet;
    ///
    /// let a: VecSet<i32> = vec![1, 2, 3, 4].into();
    /// let b: VecSet<i32> = vec![3, 4, 5, 6].into();
    ///
    /// let intersection: Vec<_> = a.intersection(&b).copied().collect();
    /// assert_eq!(intersection, vec![3, 4]);
    /// ```
    #[inline]
    pub fn intersection<'a>(&'a self, other: &'a Self) -> Intersection<'a, T> {
        Intersection {
            left: self.values.iter().peekable(),
            right: other.values.iter().peekable(),
        }
    }

    /// Returns an iterator yielding all elements from both `self` and `other`, deduplicated.
    ///
    /// The iterator borrows both sets and does not allocate. Elements are yielded in ascending
    /// order as `&T` references.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::VecSet;
    ///
    /// let a: VecSet<i32> = vec![1, 3].into();
    /// let b: VecSet<i32> = vec![2, 3].into();
    ///
    /// let union: Vec<_> = a.union(&b).copied().collect();
    /// assert_eq!(union, vec![1, 2, 3]);
    /// ```
    #[inline]
    pub fn union<'a>(&'a self, other: &'a Self) -> Union<'a, T> {
        Union {
            left: self.values.iter().peekable(),
            right: other.values.iter().peekable(),
        }
    }

    /// Retains only the elements matching the predicate.
    ///
    /// Elements for which `f` returns `false` are removed. This is O(n) in the worst case (all
    /// elements removed).
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::VecSet;
    ///
    /// let mut set: VecSet<i32> = vec![1, 2, 3, 4, 5].into();
    /// set.retain(|&x| x % 2 == 0);
    /// assert_eq!(set.len(), 2); // {2, 4}
    /// ```
    #[inline]
    pub fn retain(&mut self, f: impl FnMut(&T) -> bool) {
        self.values.retain(f);
    }
}

impl<T: Ord> Default for VecSet<T> {
    /// Creates an empty `VecSet`.
    #[inline]
    fn default() -> Self {
        VecSet::new()
    }
}

impl<T: Ord> From<Vec<T>> for VecSet<T> {
    /// Converts a `Vec<T>` into a `VecSet` by sorting and deduplicating.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::VecSet;
    ///
    /// let vec = vec![3, 1, 2, 1, 3];
    /// let set: VecSet<i32> = vec.into();
    /// assert_eq!(set.len(), 3); // {1, 2, 3}
    /// ```
    #[inline]
    fn from(mut vec: Vec<T>) -> Self {
        vec.sort();
        vec.dedup();
        VecSet { values: vec }
    }
}

impl<T: Ord, const N: usize> From<[T; N]> for VecSet<T> {
    /// Converts an array `[T; N]` into a `VecSet` by sorting and deduplicating.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::VecSet;
    ///
    /// let set: VecSet<i32> = [3, 1, 2, 1].into();
    /// assert_eq!(set.len(), 3);
    /// ```
    #[inline]
    fn from(array: [T; N]) -> Self {
        VecSet::from(Vec::from(array))
    }
}

impl<T: Ord> FromIterator<T> for VecSet<T> {
    /// Collects items from an iterator into a `VecSet`, sorting and deduplicating.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::VecSet;
    ///
    /// let set: VecSet<i32> = vec![3, 1, 2, 1].into_iter().collect();
    /// assert_eq!(set.len(), 3);
    /// ```
    #[inline]
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut vec: Vec<T> = iter.into_iter().collect();
        vec.sort();
        vec.dedup();
        VecSet { values: vec }
    }
}

pub type Iter<'a, T> = std::slice::Iter<'a, T>;
pub type IntoIter<T> = std::vec::IntoIter<T>;

impl<T: Ord> IntoIterator for VecSet<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.values.into_iter()
    }
}

impl<'a, T: Ord> IntoIterator for &'a VecSet<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.values.iter()
    }
}

impl<T: Ord> Extend<T> for VecSet<T> {
    /// Extends the set with elements from an iterator.
    ///
    /// Each element is inserted in sorted order (duplicates are silently skipped).
    #[inline]
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for value in iter {
            self.insert(value);
        }
    }

    #[cfg(feature = "unstable")]
    #[inline]
    fn extend_one(&mut self, value: T) {
        self.insert(value);
    }
}

impl<'a, T: 'a + Ord + Copy> Extend<&'a T> for VecSet<T> {
    /// Extends the set with copies of elements from an iterator of references.
    #[inline]
    fn extend<I: IntoIterator<Item = &'a T>>(&mut self, iter: I) {
        for value in iter {
            self.insert(*value);
        }
    }

    #[cfg(feature = "unstable")]
    #[inline]
    fn extend_one(&mut self, value: T) {
        self.insert(value);
    }
}

impl<T: Ord + fmt::Debug> fmt::Debug for VecSet<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_set().entries(self.values.iter()).finish()
    }
}

impl<T: Ord> ops::Deref for VecSet<T> {
    type Target = [T];

    /// Dereferences the `VecSet` to a slice, enabling indexing and slicing.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::VecSet;
    ///
    /// let set: VecSet<i32> = vec![1, 2, 3].into();
    /// assert_eq!(set[0], 1);
    /// assert_eq!(&set[..2], &[1, 2]);
    /// ```
    fn deref(&self) -> &Self::Target {
        &self.values
    }
}

/// An iterator yielding the difference between two `VecSet`s. Yields all elements that are in
/// `self` but not in `other`.
///
/// Created by [`VecSet::difference()`]. Borrows both sets and does not allocate. Elements are
/// yielded as `&T` references in ascending order.
///
/// # Example
///
/// ```
/// # use sammlung::VecSet;
///
/// let a: VecSet<i32> = vec![1, 2, 3, 4].into();
/// let b: VecSet<i32> = vec![3, 4, 5].into();
///
/// let diff: Vec<_> = a.difference(&b).copied().collect();
/// assert_eq!(diff, vec![1, 2]);
/// ```
#[derive(Debug, Clone)]
pub struct Difference<'a, T: Ord> {
    added: Iter<'a, T>,
    removed: Peekable<Iter<'a, T>>,
}

impl<'a, T: Ord> Iterator for Difference<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(removed) = self.removed.peek() {
            {
                let added = self.added.next()?;
                match added.cmp(removed) {
                    std::cmp::Ordering::Less => return Some(added),
                    std::cmp::Ordering::Equal => {
                        self.removed.next();
                    }
                    std::cmp::Ordering::Greater => {
                        self.removed.next();
                        return Some(added);
                    }
                }
            }
        }
        self.added.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let added_len = self.added.len();
        let removed_len = self.removed.len();
        (added_len.saturating_sub(removed_len), Some(added_len))
    }
}

impl<T: Ord> FusedIterator for Difference<'_, T> {}

/// An iterator yielding the symmetric difference between two `VecSet`s. Yields all elements that
/// are in either `self` or `other`, but not in both.
///
/// Created by [`VecSet::symmetric_difference()`]. Borrows both sets and does not allocate.
/// Elements are yielded as `&T` references in ascending order.
///
/// # Example
///
/// ```
/// # use sammlung::VecSet;
///
/// let a: VecSet<i32> = vec![1, 2, 3].into();
/// let b: VecSet<i32> = vec![3, 4, 5].into();
///
/// let sym_diff: Vec<_> = a.symmetric_difference(&b).copied().collect();
/// assert_eq!(sym_diff, vec![1, 2, 4, 5]);
/// ```
#[derive(Debug, Clone)]
pub struct SymmetricDifference<'a, T: Ord> {
    left: Peekable<Iter<'a, T>>,
    right: Peekable<Iter<'a, T>>,
}

impl<'a, T: Ord> Iterator for SymmetricDifference<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        while let (Some(left), Some(right)) = (self.left.peek(), self.right.peek()) {
            match left.cmp(right) {
                std::cmp::Ordering::Less => {
                    return self.left.next();
                }
                std::cmp::Ordering::Greater => {
                    self.right.next();
                }
                std::cmp::Ordering::Equal => {
                    self.left.next();
                    self.right.next();
                }
            }
        }
        if let Some(left) = self.left.next() {
            return Some(left);
        }
        self.right.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let left_len = self.left.len();
        let right_len = self.right.len();
        (
            left_len.saturating_sub(right_len),
            Some(left_len + right_len),
        )
    }
}

impl<T: Ord> FusedIterator for SymmetricDifference<'_, T> {}

/// An iterator yielding the intersection between two `VecSet`s. Yields all elements that are in
/// both `self` and `other`.
///
/// Created by [`VecSet::intersection()`]. Borrows both sets and does not allocate. Elements are
/// yielded as `&T` references in ascending order.
///
/// # Example
///
/// ```
/// # use sammlung::VecSet;
///
/// let a: VecSet<i32> = vec![1, 2, 3, 4].into();
/// let b: VecSet<i32> = vec![3, 4, 5, 6].into();
///
/// let intersection: Vec<_> = a.intersection(&b).copied().collect();
/// assert_eq!(intersection, vec![3, 4]);
/// ```
#[derive(Debug, Clone)]
pub struct Intersection<'a, T: Ord> {
    left: Peekable<Iter<'a, T>>,
    right: Peekable<Iter<'a, T>>,
}

impl<'a, T: Ord> Iterator for Intersection<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        while let (Some(left), Some(right)) = (self.left.peek(), self.right.peek()) {
            match left.cmp(right) {
                std::cmp::Ordering::Less => {
                    self.left.next();
                }
                std::cmp::Ordering::Greater => {
                    self.right.next();
                }
                std::cmp::Ordering::Equal => {
                    self.right.next();
                    return self.left.next();
                }
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let left_len = self.left.len();
        let right_len = self.right.len();
        (0, Some(left_len.min(right_len)))
    }
}

impl<T: Ord> FusedIterator for Intersection<'_, T> {}

/// An iterator yielding the union of two `VecSet`s. Yields all elements from both sets,
/// deduplicated.
///
/// Created by [`VecSet::union()`]. Borrows both sets and does not allocate. Elements are yielded
/// as `&T` references in ascending order.
///
/// # Example
///
/// ```
/// # use sammlung::VecSet;
///
/// let a: VecSet<i32> = vec![1, 3].into();
/// let b: VecSet<i32> = vec![2, 3].into();
///
/// let union: Vec<_> = a.union(&b).copied().collect();
/// assert_eq!(union, vec![1, 2, 3]);
/// ```
#[derive(Debug, Clone)]
pub struct Union<'a, T: Ord> {
    left: Peekable<Iter<'a, T>>,
    right: Peekable<Iter<'a, T>>,
}

impl<'a, T: Ord> Iterator for Union<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if let (Some(left), Some(right)) = (self.left.peek(), self.right.peek()) {
            match left.cmp(right) {
                std::cmp::Ordering::Less => {
                    return self.left.next();
                }
                std::cmp::Ordering::Greater => {
                    return self.right.next();
                }
                std::cmp::Ordering::Equal => {
                    self.right.next();
                    return self.left.next();
                }
            }
        }
        if let Some(left) = self.left.next() {
            Some(left)
        } else {
            self.right.next()
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let left_len = self.left.len();
        let right_len = self.right.len();
        (left_len.max(right_len), Some(left_len + right_len))
    }
}

impl<T: Ord> FusedIterator for Union<'_, T> {}

impl<T: Ord + Clone> ops::BitOr<&'_ VecSet<T>> for &VecSet<T> {
    type Output = VecSet<T>;

    /// The `\|` operator — returns a new `VecSet` containing the union of both sets.
    ///
    /// Equivalent to `self.union(rhs).cloned().collect()`, but allocates the result.
    fn bitor(self, rhs: &VecSet<T>) -> Self::Output {
        VecSet::from_sorted_iter(self.union(rhs).cloned())
    }
}

impl<T: Ord + Clone> ops::BitAnd<&'_ VecSet<T>> for &VecSet<T> {
    type Output = VecSet<T>;

    /// The `&` operator — returns a new `VecSet` containing the intersection of both sets.
    fn bitand(self, rhs: &VecSet<T>) -> Self::Output {
        VecSet::from_sorted_iter(self.intersection(rhs).cloned())
    }
}

impl<T: Ord + Clone> ops::BitXor<&'_ VecSet<T>> for &VecSet<T> {
    type Output = VecSet<T>;

    /// The `^` operator — returns a new `VecSet` containing the symmetric difference of both sets.
    fn bitxor(self, rhs: &VecSet<T>) -> Self::Output {
        VecSet::from_sorted_iter(self.symmetric_difference(rhs).cloned())
    }
}

impl<T: Ord + Clone> ops::Sub<&'_ VecSet<T>> for &VecSet<T> {
    type Output = VecSet<T>;

    /// The `-` operator — returns a new `VecSet` containing the difference of both sets.
    fn sub(self, rhs: &VecSet<T>) -> Self::Output {
        VecSet::from_sorted_iter(self.difference(rhs).cloned())
    }
}

impl<T: Ord + Clone> ops::BitOrAssign<&'_ VecSet<T>> for VecSet<T> {
    /// The `\|=` operator — in-place union with a borrowed set.
    fn bitor_assign(&mut self, rhs: &VecSet<T>) {
        self.union_assign_sorted(rhs.iter().cloned());
    }
}

impl<T: Ord> ops::BitOrAssign<VecSet<T>> for VecSet<T> {
    /// The `\|=` operator — in-place union with an owned set.
    ///
    /// Optimized by swapping with the RHS when it has greater capacity, then merging.
    #[inline]
    fn bitor_assign(&mut self, mut rhs: VecSet<T>) {
        if self.values.capacity() < rhs.len() {
            core::mem::swap(self, &mut rhs);
        }
        self.union_assign_sorted(rhs);
    }
}

impl<T: Ord> ops::BitAndAssign<&'_ VecSet<T>> for VecSet<T> {
    /// The `&=` operator — in-place intersection with a borrowed set.
    fn bitand_assign(&mut self, rhs: &VecSet<T>) {
        let mut current = rhs.iter().peekable();
        self.values.retain(|value| {
            while let Some(other) = current.peek() {
                match value.cmp(other) {
                    std::cmp::Ordering::Less => return false,
                    std::cmp::Ordering::Greater => {
                        current.next();
                    }
                    std::cmp::Ordering::Equal => {
                        current.next();
                        return true;
                    }
                }
            }
            false
        });
    }
}

impl<T: Ord> ops::BitAndAssign<VecSet<T>> for VecSet<T> {
    /// The `&=` operator — in-place intersection with an owned set.
    #[inline]
    fn bitand_assign(&mut self, rhs: VecSet<T>) {
        self.bitand_assign(&rhs);
    }
}

impl<T: Ord + Clone> ops::BitXorAssign<&'_ VecSet<T>> for VecSet<T> {
    /// The `^=` operator — in-place symmetric difference with a borrowed set.
    fn bitxor_assign(&mut self, rhs: &VecSet<T>) {
        self.symmetric_difference_assign_sorted(rhs.iter().cloned());
    }
}

impl<T: Ord> ops::BitXorAssign<VecSet<T>> for VecSet<T> {
    /// The `^=` operator — in-place symmetric difference with an owned set.
    ///
    /// Optimized by swapping with the RHS when it has greater capacity, then merging.
    #[inline]
    fn bitxor_assign(&mut self, mut rhs: VecSet<T>) {
        if self.values.capacity() < rhs.len() {
            core::mem::swap(self, &mut rhs);
        }
        self.symmetric_difference_assign_sorted(rhs);
    }
}

impl<T: Ord> ops::SubAssign<&'_ VecSet<T>> for VecSet<T> {
    /// The `-=` operator — in-place difference with a borrowed set.
    fn sub_assign(&mut self, rhs: &VecSet<T>) {
        let mut current = rhs.iter().peekable();
        self.values.retain(|value| {
            while let Some(other) = current.peek() {
                match value.cmp(other) {
                    std::cmp::Ordering::Less => return true,
                    std::cmp::Ordering::Greater => {
                        current.next();
                    }
                    std::cmp::Ordering::Equal => {
                        current.next();
                        return false;
                    }
                }
            }
            true
        });
    }
}

impl<T: Ord> ops::SubAssign<VecSet<T>> for VecSet<T> {
    /// The `-=` operator — in-place difference with an owned set.
    #[inline]
    fn sub_assign(&mut self, rhs: VecSet<T>) {
        self.sub_assign(&rhs);
    }
}
