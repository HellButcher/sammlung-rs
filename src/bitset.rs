use core::{fmt, ptr::NonNull};
use std::{alloc, cmp, hash::Hash, iter::FusedIterator, marker::PhantomData, ops};

type Word = u128;

#[cfg(feature = "unstable")]
use alloc::{Allocator, Global};
#[cfg(not(feature = "unstable"))]
use alloc::{GlobalAlloc as Allocator, System as Global};

/// A bit-packed set backed by a manually managed `Vec<u128>`. Each bit position `i` represents
/// whether the value `i` is present.
///
/// No map payload — this is a pure set (boolean presence). Insert, remove, and contains are all
/// O(1) with excellent cache locality and minimal memory: one bit per possible value.
///
/// # Best for
///
/// Dense ranges of non-negative integers where the maximum value fits in available memory. For
/// example, tracking user IDs in the range 0–1,000,000 uses only ~125 KB.
///
/// # Generic allocator
///
/// `BitSet` is generic over an [`Allocator`] (defaulting to the system allocator). This allows
/// constructing bit sets with custom allocators via [`new_in`](BitSet::new_in) and
/// [`with_capacity_in`](BitSet::with_capacity_in).
///
/// # Example
///
/// ```
/// # use sammlung::BitSet;
///
/// let mut set = BitSet::with_capacity(100);
/// set.insert(5);
/// set.insert(42);
/// assert!(set.contains(5));
/// assert!(!set.contains(7));
/// assert_eq!(set.len(), 2);
/// ```
///
/// # Set operations
///
/// `BitSet` supports standard set operations via both method calls and operator overloads:
///
/// | Operation | Method | Operator |
/// |---|---|---|
/// | Union | `.union(other)` | `a | b` |
/// | Intersection | `.intersection(other)` | `a & b` |
/// | Difference | `.difference(other)` | `a - b` |
/// | Symmetric difference | `.symmetric_difference(other)` | `a ^ b` |
///
/// The iterator-based methods return zero-allocation iterators that borrow both sets. The operator
/// variants (`|=`, `&=`, `^=`, `-=`) mutate `self` in place.
///
/// # Memory layout
///
/// Internally, each `u128` word holds 128 bits. Capacity grows geometrically (powers of two) to
/// amortize reallocation cost. The `capacity()` method reports the number of *values* that fit, not
/// the number of words.
pub struct BitSet<A: Allocator = Global> {
    bits: NonNull<Word>,
    capacity_words: usize,
    alloc: A,
}

impl BitSet {
    /// Create a new empty `BitSet` using the default system allocator.
    ///
    /// This is a `const fn` and can be called in const contexts. For a pre-sized set, use
    /// [`with_capacity`](BitSet::with_capacity) instead.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::BitSet;
    ///
    /// let set = BitSet::new();
    /// assert!(set.is_empty());
    /// ```
    pub const fn new() -> Self {
        Self::new_in(Global)
    }

    /// Create a new empty `BitSet` with capacity for at least `num_bits` values.
    ///
    /// The internal buffer is allocated up front to hold at least `num_bits` set bits, avoiding
    /// reallocations during subsequent insertions.
    ///
    /// # Arguments
    ///
    /// * `num_bits` — The minimum number of distinct values that can be stored.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::BitSet;
    ///
    /// let mut set = BitSet::with_capacity(1000);
    /// for i in 0..1000 {
    ///     set.insert(i);
    /// }
    /// assert_eq!(set.len(), 1000);
    /// ```
    #[inline]
    pub fn with_capacity(num_bits: usize) -> Self {
        Self::with_capacity_in(num_bits, Global)
    }
}

impl<A: Allocator> BitSet<A> {
    /// Create a new empty `BitSet` using the provided allocator.
    ///
    /// This is a `const fn` and can be called in const contexts.
    ///
    /// # Arguments
    ///
    /// * `alloc` — The allocator to use for internal storage.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::BitSet;
    /// # use std::alloc::System;
    ///
    /// let set = BitSet::new_in(System);
    /// assert!(set.is_empty());
    /// ```
    #[inline]
    pub const fn new_in(alloc: A) -> Self {
        BitSet {
            bits: NonNull::dangling(),
            capacity_words: 0,
            alloc,
        }
    }

    /// Create a new empty `BitSet` with the provided allocator and capacity for at least
    /// `num_bits` values.
    ///
    /// # Arguments
    ///
    /// * `num_bits` — The minimum number of distinct values that can be stored.
    /// * `alloc` — The allocator to use for internal storage.
    pub fn with_capacity_in(num_bits: usize, alloc: A) -> Self {
        let mut result = Self::new_in(alloc);
        if num_bits > 0 {
            let words = get_num_words(num_bits);
            result.resize(words);
        }
        result
    }

    /// Returns the number of values (bits) that can be stored without reallocation.
    ///
    /// This is the allocated capacity measured in *values*, not in words. For example, if the
    /// internal buffer holds 8 `u128` words, `capacity()` returns `8 * 128 = 1024`.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::BitSet;
    ///
    /// let set = BitSet::with_capacity(500);
    /// assert!(set.capacity() >= 500);
    /// ```
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity_words << Word::BITS.ilog2()
    }

    #[inline]
    fn resize(&mut self, new_capacity_words: usize) {
        let old_capacity = self.capacity_words;
        if old_capacity == new_capacity_words {
        } else if new_capacity_words == 0 {
            // deallocate
            let old_layout = alloc::Layout::array::<Word>(old_capacity).unwrap();
            let old_ptr = self.bits.as_ptr().cast();
            self.bits = NonNull::dangling();
            self.capacity_words = 0;
            unsafe {
                self.alloc.dealloc(old_ptr, old_layout);
            }
        } else if old_capacity == 0 {
            // allocate
            let new_layout = alloc::Layout::array::<Word>(new_capacity_words).unwrap();
            let new_bits = unsafe {
                NonNull::new(self.alloc.alloc_zeroed(new_layout).cast())
                    .unwrap_or_else(|| alloc::handle_alloc_error(new_layout))
            };
            self.bits = new_bits;
            self.capacity_words = new_capacity_words;
        } else {
            // reallocate
            let old_layout = alloc::Layout::array::<Word>(old_capacity).unwrap();
            let new_layout = alloc::Layout::array::<Word>(new_capacity_words).unwrap();
            let old_ptr = self.bits.as_ptr().cast();
            let new_bits: NonNull<Word> = unsafe {
                NonNull::new(
                    self.alloc
                        .realloc(old_ptr, old_layout, new_layout.size())
                        .cast(),
                )
                .unwrap_or_else(|| alloc::handle_alloc_error(new_layout))
            };
            unsafe {
                for i in old_capacity..new_capacity_words {
                    new_bits.as_ptr().add(i).write(0);
                }
            }
            self.bits = new_bits;
            self.capacity_words = new_capacity_words;
        }
    }

    /// Shrink the internal buffer to fit the current number of set bits.
    ///
    /// The behavior of this method depends on the allocator, which may either shrink the buffer in-place or reallocate.
    pub fn shrink_to_fit(&mut self) {
        let trimmed_words = self.words_trim();
        let new_capacity = trimmed_words.len();
        if new_capacity < self.capacity_words {
            self.resize(new_capacity);
        }
    }

    #[inline]
    fn words(&self) -> &[Word] {
        unsafe { std::slice::from_raw_parts(self.bits.as_ptr(), self.capacity_words) }
    }

    #[inline]
    fn words_mut(&mut self) -> &mut [Word] {
        unsafe { std::slice::from_raw_parts_mut(self.bits.as_ptr(), self.capacity_words) }
    }

    #[inline]
    fn word(&self, word_idx: usize) -> Word {
        if word_idx >= self.capacity_words {
            0
        } else {
            unsafe { *self.bits.as_ptr().add(word_idx) }
        }
    }

    fn words_trim(&self) -> &[Word] {
        let words = self.words();
        for (i, word) in words.iter().enumerate().rev() {
            if *word != 0 {
                return &words[..=i];
            }
        }
        &[]
    }

    #[inline]
    fn word_mut(&mut self, word_idx: usize) -> &mut Word {
        if word_idx >= self.capacity_words {
            // Use (word_idx + 1).next_power_of_two() to ensure the new capacity
            // is strictly greater than word_idx. If word_idx is already a power
            // of two, next_power_of_two(word_idx) == word_idx, which would give
            // a no-op resize and leave us one word short.
            self.resize((word_idx + 1).next_power_of_two());
        }
        unsafe { &mut *self.bits.as_ptr().add(word_idx) }
    }

    /// Set bit at position `value`.
    ///
    /// Returns `true` if the bit was **already** set (i.e., this is a no-op idempotent insertion),
    /// and `false` if the bit was previously unset and has now been set.
    ///
    /// This operation is O(1). If the internal buffer is too small, it grows geometrically.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::BitSet;
    ///
    /// let mut set = BitSet::new();
    /// assert!(!set.insert(42));  // newly inserted
    /// assert!(set.insert(42));   // already present
    /// assert!(set.contains(42));
    /// ```
    #[inline]
    pub fn insert(&mut self, value: usize) -> bool {
        let (word_idx, bit_idx) = decompose(value);
        let word = self.word_mut(word_idx);
        let mask: Word = 1 << bit_idx;
        let was_set = *word & mask != 0;
        *word |= mask;
        was_set
    }

    /// Clear bit at position `value` if present.
    ///
    /// Returns `true` if the bit was previously set and is now cleared, and `false` if the bit was
    /// already unset (no change).
    ///
    /// This operation is O(1). If the value is beyond the current capacity, it returns `false`.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::BitSet;
    ///
    /// let mut set = BitSet::new();
    /// set.insert(10);
    /// assert!(set.remove(10));   // was present, now removed
    /// assert!(!set.remove(10));  // was not present
    /// assert!(!set.contains(10));
    /// ```
    #[inline]
    pub fn remove(&mut self, value: usize) -> bool {
        let (word_idx, bit_idx) = decompose(value);
        if word_idx >= self.capacity_words {
            return false;
        }
        let word = self.word_mut(word_idx);
        let mask: Word = 1 << bit_idx;
        let was_set = *word & mask != 0;
        *word &= !mask;
        was_set
    }

    /// Clear all bits in the `BitSet`, keeping the internal allocation for reuse.
    ///
    /// After calling `clear()`, [`is_empty()`](BitSet::is_empty) returns `true` and [`len()`](BitSet::len) returns 0,
    /// but the capacity remains unchanged so subsequent insertions do not trigger reallocation.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::BitSet;
    ///
    /// let mut set = BitSet::with_capacity(1000);
    /// for i in 0..500 { set.insert(i); }
    /// assert_eq!(set.len(), 500);
    ///
    /// set.clear();
    /// assert!(set.is_empty());
    /// assert_eq!(set.len(), 0);
    /// // capacity is preserved
    /// ```
    #[inline]
    pub fn clear(&mut self) {
        for word in self.words_mut() {
            *word = 0;
        }
    }

    /// Check if the bit at position `value` is set.
    ///
    /// Returns `true` if value `value` is present in the set, `false` otherwise.
    ///
    /// This operation is O(1). If the value is beyond the current capacity, returns `false`.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::BitSet;
    ///
    /// let mut set = BitSet::new();
    /// set.insert(7);
    /// assert!(set.contains(7));
    /// assert!(!set.contains(8));
    /// ```
    #[inline]
    pub fn contains(&self, value: usize) -> bool {
        let (word_idx, bit_idx) = decompose(value);
        if word_idx >= self.capacity_words {
            return false;
        }
        (self.word(word_idx) & (1 << bit_idx)) != 0
    }

    /// Returns the number of set bits in the `BitSet`.
    ///
    /// HINT: this is /O(N)/: it counts the set bits in all words.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::BitSet;
    ///
    /// let mut set = BitSet::new();
    /// assert_eq!(set.len(), 0);
    ///
    /// set.insert(1);
    /// set.insert(3);
    /// set.insert(5);
    /// assert_eq!(set.len(), 3);
    /// ```
    #[inline]
    pub fn len(&self) -> usize {
        self.words().iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Returns `true` if the `BitSet` contains no set bits.
    ///
    /// HINT: this is /O(N)/: it counts the set bits in all words.
    /// # Example
    ///
    /// ```
    /// # use sammlung::BitSet;
    ///
    /// let mut set = BitSet::new();
    /// assert!(set.is_empty());
    ///
    /// set.insert(0);
    /// assert!(!set.is_empty());
    /// ```
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the smallest set bit, or `None` if the set is empty.
    ///
    /// This scans from the lowest-indexed word to find the first bit that is set, then returns its
    /// position as a `usize`.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::BitSet;
    ///
    /// let mut set = BitSet::new();
    /// assert!(set.first().is_none());
    ///
    /// set.insert(100);
    /// set.insert(42);
    /// set.insert(7);
    /// assert_eq!(set.first(), Some(7));
    /// ```
    pub fn first(&self) -> Option<usize> {
        for (word_idx, word) in self.words().iter().enumerate() {
            if *word != 0 {
                let bit_idx = word.trailing_zeros();
                return Some(compose(word_idx, bit_idx));
            }
        }
        None
    }

    /// Returns the largest set bit, or `None` if the set is empty.
    ///
    /// This scans from the highest-indexed word to find the first bit that is set, then returns its
    /// position as a `usize`.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::BitSet;
    ///
    /// let mut set = BitSet::new();
    /// assert!(set.last().is_none());
    ///
    /// set.insert(10);
    /// set.insert(100);
    /// set.insert(42);
    /// assert_eq!(set.last(), Some(100));
    /// ```
    pub fn last(&self) -> Option<usize> {
        for (word_idx, word) in self.words().iter().enumerate().rev() {
            if *word != 0 {
                let bit_idx = Word::BITS - 1 - word.leading_zeros();
                return Some(compose(word_idx, bit_idx));
            }
        }
        None
    }

    /// Returns an iterator over the set bits in ascending order.
    ///
    /// The iterator yields each value whose bit is set, starting from the smallest. It implements
    /// [`FusedIterator`], so calling [`next()`](Iterator::next) after it is exhausted always returns `None`.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::BitSet;
    ///
    /// let mut set = BitSet::new();
    /// set.insert(5);
    /// set.insert(1);
    /// set.insert(9);
    ///
    /// let values: Vec<_> = set.iter().collect();
    /// assert_eq!(values, vec![1, 5, 9]);
    /// ```
    #[inline]
    pub fn iter(&self) -> Iter<'_> {
        Iter {
            words: self.words_trim(),
            current_word: 0,
            next_word_index: 0,
        }
    }

    /// Returns an iterator yielding all bits set in either `self` or `other` (or both).
    ///
    /// The iterator borrows both sets and does not allocate. Elements are yielded in ascending
    /// order. Use [`MergedIter::rest_as_set`] to collect remaining unconsumed elements.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::BitSet;
    ///
    /// let mut a = BitSet::new();
    /// a.insert(1); a.insert(3); a.insert(5);
    ///
    /// let mut b = BitSet::new();
    /// b.insert(2); b.insert(3); b.insert(6);
    ///
    /// let union: Vec<_> = a.union(&b).collect();
    /// assert_eq!(union, vec![1, 2, 3, 5, 6]);
    /// ```
    #[inline]
    pub fn union<'a>(&'a self, other: &'a BitSet<impl Allocator>) -> Union<'a> {
        Union::new(self.words_trim(), other.words_trim())
    }

    /// Returns an iterator yielding bits set in both `self` and `other`.
    ///
    /// The iterator borrows both sets and does not allocate. Elements are yielded in ascending
    /// order.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::BitSet;
    ///
    /// let mut a = BitSet::new();
    /// a.insert(1); a.insert(3); a.insert(5);
    ///
    /// let mut b = BitSet::new();
    /// b.insert(3); b.insert(5); b.insert(7);
    ///
    /// let intersection: Vec<_> = a.intersection(&b).collect();
    /// assert_eq!(intersection, vec![3, 5]);
    /// ```
    #[inline]
    pub fn intersection<'a>(&'a self, other: &'a BitSet<impl Allocator>) -> Intersection<'a> {
        Intersection::new(self.words_trim(), other.words())
    }

    /// Returns an iterator yielding bits set in `self` but not in `other`.
    ///
    /// The iterator borrows both sets and does not allocate. Elements are yielded in ascending
    /// order.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::BitSet;
    ///
    /// let mut a = BitSet::new();
    /// a.insert(1); a.insert(3); a.insert(5);
    ///
    /// let mut b = BitSet::new();
    /// b.insert(3); b.insert(5); b.insert(7);
    ///
    /// let diff: Vec<_> = a.difference(&b).collect();
    /// assert_eq!(diff, vec![1]);
    /// ```
    #[inline]
    pub fn difference<'a>(&'a self, other: &'a BitSet<impl Allocator>) -> Difference<'a> {
        Difference::new(self.words_trim(), other.words())
    }

    /// Returns an iterator yielding bits set in exactly one of `self` or `other` (but not both).
    ///
    /// The iterator borrows both sets and does not allocate. Elements are yielded in ascending
    /// order.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::BitSet;
    ///
    /// let mut a = BitSet::new();
    /// a.insert(1); a.insert(3); a.insert(5);
    ///
    /// let mut b = BitSet::new();
    /// b.insert(3); b.insert(5); b.insert(7);
    ///
    /// let sym_diff: Vec<_> = a.symmetric_difference(&b).collect();
    /// assert_eq!(sym_diff, vec![1, 7]);
    /// ```
    #[inline]
    pub fn symmetric_difference<'a>(
        &'a self,
        other: &'a BitSet<impl Allocator>,
    ) -> SymmetricDifference<'a> {
        SymmetricDifference::new(self.words_trim(), other.words_trim())
    }

    fn apply_merge<M: sealed::MergeOp>(&mut self, other: &[Word]) {
        let a_len = self.capacity_words;
        let mut idx = 0;
        loop {
            let word_a = if idx < a_len {
                Some(self.word(idx))
            } else {
                None
            };
            let word_b = other.get(idx).copied();
            match M::merge(word_a, word_b) {
                Some(new_word) => {
                    *self.word_mut(idx) = new_word;
                    idx += 1;
                }
                None => break,
            }
        }
        while idx < a_len {
            *self.word_mut(idx) = 0;
            idx += 1;
        }
    }
}

impl<A: Allocator + Default> Default for BitSet<A> {
    /// Creates an empty `BitSet` using a default-constructed allocator.
    fn default() -> Self {
        Self::new_in(A::default())
    }
}

impl<A: Allocator> Drop for BitSet<A> {
    fn drop(&mut self) {
        if self.capacity_words != 0 {
            self.resize(0); // deallocate
        }
    }
}

impl<A: Allocator + Clone> Clone for BitSet<A> {
    fn clone(&self) -> Self {
        let mut new_set = Self::new_in(self.alloc.clone());
        if let Some(last) = self.last() {
            new_set.resize(get_num_words(last + 1));
            unsafe {
                core::ptr::copy_nonoverlapping(
                    self.bits.as_ptr(),
                    new_set.bits.as_ptr(),
                    self.capacity_words,
                );
            }
        }
        new_set
    }
}

/// An iterator over the set bits of a [`BitSet`], yielding `usize` values in ascending order.
///
/// Created by [`BitSet::iter()`]. Implements [`FusedIterator`] — calling [`next()`](Iterator::next)
/// after exhaustion always returns `None`.
///
/// # Example
///
/// ```
/// # use sammlung::BitSet;
///
/// let mut set = BitSet::new();
/// set.insert(3);
/// set.insert(1);
/// set.insert(7);
///
/// for value in set.iter() {
///     println!("{value}");  // prints 1, 3, 7 in order
/// }
/// ```
#[derive(Copy, Clone)]
pub struct Iter<'a> {
    words: &'a [Word],
    current_word: Word,
    next_word_index: usize,
}

impl Iterator for Iter<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        while self.current_word == 0 {
            let word = self.words.get(self.next_word_index)?;
            self.current_word = *word;
            self.next_word_index += 1;
        }
        let bit_idx = self.current_word.trailing_zeros();
        self.current_word &= !(1 << bit_idx);
        let result = compose(self.next_word_index - 1, bit_idx);
        Some(result)
    }
}

impl FusedIterator for Iter<'_> {}

impl<'a> IntoIterator for &'a BitSet {
    type Item = usize;
    type IntoIter = Iter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        Iter {
            words: self.words(),
            current_word: 0,
            next_word_index: 0,
        }
    }
}

/// A generic merged iterator over two word slices, used as the base for all `BitSet` set-diff
/// iterators.
///
/// This struct is parameterized by a sealed merge operation that defines how corresponding words from
/// the two input slices are combined. It is used internally by [`Union`], [`Intersection`],
/// [`Difference`], and [`SymmetricDifference`].
///
/// # Reuse
///
/// Call [`reset()`](MergedIter::reset) to rewind the iterator to the beginning, or
/// [`rest_as_set()`](MergedIter::rest_as_set) to collect remaining unconsumed elements into a new
/// `BitSet`.
///
/// # Example
///
/// ```
/// # use sammlung::BitSet;
///
/// let mut a = BitSet::new();
/// a.insert(1); a.insert(3); a.insert(5);
///
/// let mut b = BitSet::new();
/// b.insert(2); b.insert(4); b.insert(6);
///
/// let iter = a.union(&b);
/// assert_eq!(iter.count(), 6);
/// ```
#[derive(Copy, Clone)]
pub struct MergedIter<'a, M> {
    slice_a: &'a [Word],
    slice_b: &'a [Word],
    current_word: Word,
    next_word_index: usize,
    _merge_fn: PhantomData<M>,
}

impl<M: sealed::MergeOp> MergedIter<'_, M> {
    /// Reset the iterator to the beginning, allowing it to be reused.
    ///
    /// After calling `reset()`, the iterator will start yielding elements from the beginning of the
    /// merged sequence again. The current partial word state is cleared.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::BitSet;
    ///
    /// let mut a = BitSet::new();
    /// a.insert(1); a.insert(2);
    ///
    /// let mut b = BitSet::new();
    /// b.insert(3);
    ///
    /// let mut iter = a.union(&b);
    /// assert_eq!(iter.next(), Some(1));
    /// iter.reset();
    /// assert_eq!(iter.next(), Some(1)); // restarted from beginning
    /// ```
    #[inline]
    pub fn reset(&mut self) {
        self.current_word = 0;
        self.next_word_index = 0;
    }

    /// Collect all remaining (unconsumed) elements into a new `BitSet` with the default allocator.
    ///
    /// Elements that have already been yielded by the iterator are excluded. The partial word state
    /// (if any) is included in the result.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::BitSet;
    ///
    /// let mut a = BitSet::new();
    /// a.insert(1); a.insert(2); a.insert(3);
    ///
    /// let mut b = BitSet::new();
    /// b.insert(2); b.insert(4);
    ///
    /// let mut iter = a.union(&b);
    /// assert_eq!(iter.next(), Some(1)); // consume first element
    ///
    /// let rest = iter.rest_as_set();
    /// assert_eq!(rest.len(), 3); // {2, 3, 4} remain
    /// ```
    pub fn rest_as_set(&self) -> BitSet {
        self.rest_as_set_in(Global)
    }

    /// Collect all remaining (unconsumed) elements into a new `BitSet` with the given allocator.
    ///
    /// Elements that have already been yielded by the iterator are excluded. The partial word state
    /// (if any) is included in the result.
    ///
    /// # Arguments
    ///
    /// * `alloc` — The allocator to use for the resulting `BitSet`.
    ///
    /// # Example
    ///
    /// ```
    /// # use sammlung::BitSet;
    /// # use std::alloc::System;
    ///
    /// let mut a = BitSet::new();
    /// a.insert(1); a.insert(2); a.insert(3);
    ///
    /// let mut b = BitSet::new();
    /// b.insert(4);
    ///
    /// let iter = a.union(&b);
    /// let rest = iter.rest_as_set_in(System);
    /// assert_eq!(rest.len(), 4);
    /// ```
    pub fn rest_as_set_in<A: Allocator>(&self, alloc: A) -> BitSet<A> {
        let mut result = BitSet::with_capacity_in(
            M::guess_needed_capacity(self.slice_a.len(), self.slice_b.len()) * Word::BITS as usize,
            alloc,
        );
        let mut current = self.next_word_index;
        if current > 0 {
            *result.word_mut(current - 1) = self.current_word;
        }
        while let Some(word) = M::merge(
            self.slice_a.get(current).copied(),
            self.slice_b.get(current).copied(),
        ) {
            *result.word_mut(current) = word;
            current += 1;
        }
        result
    }
}

mod sealed {
    use std::{iter::FusedIterator, marker::PhantomData};

    use super::{MergedIter, Word};

    impl<'a, M> MergedIter<'a, M> {
        pub fn new(slice_a: &'a [Word], slice_b: &'a [Word]) -> Self {
            Self {
                slice_a,
                slice_b,
                current_word: 0,
                next_word_index: 0,
                _merge_fn: PhantomData,
            }
        }
    }

    impl<'a, M> Iterator for MergedIter<'a, M>
    where
        M: MergeOp,
    {
        type Item = usize;

        fn next(&mut self) -> Option<Self::Item> {
            while self.current_word == 0 {
                let word_a = self.slice_a.get(self.next_word_index).copied();
                let word_b = self.slice_b.get(self.next_word_index).copied();
                self.current_word = M::merge(word_a, word_b)?;
                self.next_word_index += 1;
            }
            let bit_idx = self.current_word.trailing_zeros();
            self.current_word &= !(1 << bit_idx);
            let result = super::compose(self.next_word_index - 1, bit_idx);
            Some(result)
        }
    }

    impl<M> FusedIterator for MergedIter<'_, M> where M: MergeOp {}

    pub trait MergeOp {
        fn merge(word_a: Option<Word>, word_b: Option<Word>) -> Option<Word>;
        fn guess_needed_capacity(len_a: usize, len_b: usize) -> usize;
    }

    pub struct UnionOp;
    impl MergeOp for UnionOp {
        fn merge(word_a: Option<Word>, word_b: Option<Word>) -> Option<Word> {
            match (word_a, word_b) {
                (Some(a), Some(b)) => Some(a | b),
                (Some(a), None) => Some(a),
                (None, Some(b)) => Some(b),
                (None, None) => None,
            }
        }
        fn guess_needed_capacity(len_a: usize, len_b: usize) -> usize {
            len_a.max(len_b)
        }
    }

    pub struct IntersectionOp;
    impl MergeOp for IntersectionOp {
        fn merge(word_a: Option<Word>, word_b: Option<Word>) -> Option<Word> {
            match (word_a, word_b) {
                (Some(a), Some(b)) => Some(a & b),
                _ => None,
            }
        }
        fn guess_needed_capacity(len_a: usize, len_b: usize) -> usize {
            len_a.min(len_b)
        }
    }

    pub struct DifferenceOp;
    impl MergeOp for DifferenceOp {
        fn merge(word_a: Option<Word>, word_b: Option<Word>) -> Option<Word> {
            match (word_a, word_b) {
                (Some(a), Some(b)) => Some(a & !b),
                (Some(a), None) => Some(a),
                _ => None,
            }
        }
        fn guess_needed_capacity(len_a: usize, _len_b: usize) -> usize {
            len_a
        }
    }

    pub struct SymmetricDifferenceOp;
    impl MergeOp for SymmetricDifferenceOp {
        fn merge(word_a: Option<Word>, word_b: Option<Word>) -> Option<Word> {
            match (word_a, word_b) {
                (Some(a), Some(b)) => Some(a ^ b),
                (Some(a), None) => Some(a),
                (None, Some(b)) => Some(b),
                (None, None) => None,
            }
        }
        fn guess_needed_capacity(len_a: usize, len_b: usize) -> usize {
            len_a.max(len_b)
        }
    }
}

/// An iterator yielding the union of two [`BitSet`]s — all bits set in either `self` or `other`.
///
/// This is a type alias for `MergedIter<'a, UnionOp>`. See [`BitSet::union()`] for details and
/// examples.
///
/// # Example
///
/// ```
/// # use sammlung::BitSet;
///
/// let mut a = BitSet::new(); a.insert(1); a.insert(2);
/// let mut b = BitSet::new(); b.insert(2); b.insert(3);
///
/// let union: Vec<_> = BitSet::union(&a, &b).collect();
/// assert_eq!(union, vec![1, 2, 3]);
/// ```
pub type Union<'a> = MergedIter<'a, sealed::UnionOp>;

/// An iterator yielding the intersection of two [`BitSet`]s — bits set in both `self` and `other`.
///
/// This is a type alias for `MergedIter<'a, IntersectionOp>`. See [`BitSet::intersection()`] for
/// details and examples.
pub type Intersection<'a> = MergedIter<'a, sealed::IntersectionOp>;

/// An iterator yielding the difference of two [`BitSet`]s — bits set in `self` but not in `other`.
///
/// This is a type alias for `MergedIter<'a, DifferenceOp>`. See [`BitSet::difference()`] for
/// details and examples.
pub type Difference<'a> = MergedIter<'a, sealed::DifferenceOp>;

/// An iterator yielding the symmetric difference of two [`BitSet`]s — bits set in exactly one of
/// `self` or `other`.
///
/// This is a type alias for `MergedIter<'a, SymmetricDifferenceOp>`. See
/// [`BitSet::symmetric_difference()`] for details and examples.
pub type SymmetricDifference<'a> = MergedIter<'a, sealed::SymmetricDifferenceOp>;

impl<'a, A: Allocator, B: Allocator> ops::BitAnd<&'a BitSet<B>> for &'a BitSet<A> {
    /// The `&` operator — returns an iterator yielding the intersection of two sets.
    ///
    /// Equivalent to calling [`intersection()`](BitSet::intersection).
    type Output = Intersection<'a>;

    fn bitand(self, rhs: &'a BitSet<B>) -> Self::Output {
        self.intersection(rhs)
    }
}

impl<'a, A: Allocator, B: Allocator> ops::BitOr<&'a BitSet<B>> for &'a BitSet<A> {
    /// The `\|` operator — returns an iterator yielding the union of two sets.
    ///
    /// Equivalent to calling [`union()`](BitSet::union).
    type Output = Union<'a>;

    fn bitor(self, rhs: &'a BitSet<B>) -> Self::Output {
        self.union(rhs)
    }
}

impl<'a, A: Allocator, B: Allocator> ops::BitXor<&'a BitSet<B>> for &'a BitSet<A> {
    /// The `^` operator — returns an iterator yielding the symmetric difference of two sets.
    ///
    /// Equivalent to calling [`symmetric_difference()`](BitSet::symmetric_difference).
    type Output = SymmetricDifference<'a>;

    fn bitxor(self, rhs: &'a BitSet<B>) -> Self::Output {
        self.symmetric_difference(rhs)
    }
}

impl<'a, A: Allocator, B: Allocator> ops::Sub<&'a BitSet<B>> for &'a BitSet<A> {
    /// The `-` operator — returns an iterator yielding the difference of two sets.
    ///
    /// Equivalent to calling [`difference()`](BitSet::difference).
    type Output = Difference<'a>;

    fn sub(self, rhs: &'a BitSet<B>) -> Self::Output {
        self.difference(rhs)
    }
}

impl<A: Allocator, B: Allocator> ops::BitAndAssign<&BitSet<B>> for BitSet<A> {
    /// The `&=` operator — in-place intersection.
    ///
    /// Removes all bits from `self` that are not also set in `rhs`. Equivalent to
    /// `*self = self.intersection(rhs).rest_as_set()`, but more efficient.
    fn bitand_assign(&mut self, rhs: &BitSet<B>) {
        self.apply_merge::<sealed::IntersectionOp>(rhs.words());
    }
}

impl<A: Allocator, B: Allocator> ops::BitOrAssign<&BitSet<B>> for BitSet<A> {
    /// The `\|=` operator — in-place union.
    ///
    /// Sets all bits from `rhs` into `self`. Equivalent to `*self |= other`, but more efficient.
    fn bitor_assign(&mut self, rhs: &BitSet<B>) {
        self.apply_merge::<sealed::UnionOp>(rhs.words());
    }
}

impl<A: Allocator, B: Allocator> ops::BitXorAssign<&BitSet<B>> for BitSet<A> {
    /// The `^=` operator — in-place symmetric difference.
    ///
    /// Toggles bits that are set in either `self` or `rhs` but not both. Equivalent to
    /// `*self = self.symmetric_difference(rhs).rest_as_set()`, but more efficient.
    fn bitxor_assign(&mut self, rhs: &BitSet<B>) {
        self.apply_merge::<sealed::SymmetricDifferenceOp>(rhs.words());
    }
}

impl<A: Allocator, B: Allocator> ops::SubAssign<&BitSet<B>> for BitSet<A> {
    /// The `-=` operator — in-place difference.
    ///
    /// Removes from `self` all bits that are set in `rhs`. Equivalent to
    /// `*self = self.difference(rhs).rest_as_set()`, but more efficient.
    fn sub_assign(&mut self, rhs: &BitSet<B>) {
        self.apply_merge::<sealed::DifferenceOp>(rhs.words());
    }
}

impl<A: Allocator> ops::BitAndAssign<Self> for BitSet<A> {
    /// The `&=` operator with an owned RHS — in-place intersection.
    fn bitand_assign(&mut self, rhs: Self) {
        self.apply_merge::<sealed::IntersectionOp>(rhs.words());
    }
}

impl<A: Allocator> ops::BitOrAssign for BitSet<A> {
    /// The `\|=` operator with an owned RHS — in-place union.
    ///
    /// Optimized by swapping with the RHS when the RHS has greater capacity, then merging.
    fn bitor_assign(&mut self, mut rhs: Self) {
        if self.capacity_words < rhs.capacity_words {
            std::mem::swap(self, &mut rhs);
        }
        self.apply_merge::<sealed::UnionOp>(rhs.words());
    }
}

impl<A: Allocator> ops::BitXorAssign for BitSet<A> {
    /// The `^=` operator with an owned RHS — in-place symmetric difference.
    ///
    /// Optimized by swapping with the RHS when the RHS has greater capacity, then merging.
    fn bitxor_assign(&mut self, mut rhs: Self) {
        if self.capacity_words < rhs.capacity_words {
            std::mem::swap(self, &mut rhs);
        }
        self.apply_merge::<sealed::SymmetricDifferenceOp>(rhs.words());
    }
}

impl<A: Allocator> ops::SubAssign for BitSet<A> {
    /// The `-=` operator with an owned RHS — in-place difference.
    fn sub_assign(&mut self, rhs: Self) {
        self.apply_merge::<sealed::DifferenceOp>(rhs.words());
    }
}

impl<A: Allocator> fmt::Debug for BitSet<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

impl<A: Allocator, B: Allocator> cmp::PartialEq<BitSet<B>> for BitSet<A> {
    fn eq(&self, other: &BitSet<B>) -> bool {
        let self_words = self.words_trim();
        let other_words = other.words_trim();
        self_words == other_words
    }
}

impl Eq for BitSet {}

impl Hash for BitSet {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        for word in self.words_trim() {
            word.hash(state);
        }
    }
}

#[inline]
const fn get_num_words(num_bits: usize) -> usize {
    get_word_index(num_bits + Word::BITS as usize - 1)
}

#[inline]
const fn get_word_index(value: usize) -> usize {
    value >> Word::BITS.ilog2()
}

#[inline]
const fn get_bit_index(value: usize) -> u32 {
    value as u32 & (Word::BITS - 1)
}

#[inline]
const fn decompose(value: usize) -> (usize, u32) {
    (get_word_index(value), get_bit_index(value))
}

#[inline]
const fn compose(word_idx: usize, bit_idx: u32) -> usize {
    (word_idx << Word::BITS.ilog2()) | bit_idx as usize
}
