# Cyclic Sort

## Explanation

The cyclic sort pattern is designed for problems involving arrays where numbers are in a specific range, usually from 1 to n. The idea is to place each number at its correct index by continuously swapping elements until they're in the right position. This pattern is particularly useful when dealing with missing or duplicate numbers in an array.

## Algorithm (Word-based)

- Iterate through the array
- For each element, if it's not at its correct index, swap it to the correct position
- The correct position for value x is index x-1
- Continue until the element is in the right position or we find a duplicate
- After all elements are in correct positions, identify missing/duplicates

## Code Template

```python
def cyclic_sort(nums):
    """
    Sort array where elements are in range [1, n] using cyclic sort
    """
    i = 0
    while i < len(nums):
        correct_pos = nums[i] - 1  # Correct position for nums[i]
        
        # If current element is not at its correct position, swap
        if nums[i] != nums[correct_pos]:
            nums[i], nums[correct_pos] = nums[correct_pos], nums[i]
        else:
            i += 1  # Move to next element if current is in correct position
    
    return nums
```

## Practice Questions

1. [Find the Missing Number](https://leetcode.com/problems/missing-number/) - Easy
2. [Find All Numbers Disappeared in an Array](https://leetcode.com/problems/find-all-numbers-disappeared-in-an-array/) - Easy
3. [Find the Duplicate Number](https://leetcode.com/problems/find-the-duplicate-number/) - Medium
4. [First Missing Positive](https://leetcode.com/problems/first-missing-positive/) - Hard
5. [Set Mismatch](https://leetcode.com/problems/set-mismatch/) - Easy
6. [Find All Duplicates in an Array](https://leetcode.com/problems/find-all-duplicates-in-an-array/) - Medium
7. [Cyclic Sort](https://leetcode.com/problems/sort-array-by-moving-items-to-empty-space/) - Medium
8. [Kth Missing Positive Number](https://leetcode.com/problems/kth-missing-positive-number/) - Easy
9. [Maximum Swap](https://leetcode.com/problems/maximum-swap/) - Medium
10. [Array Nesting](https://leetcode.com/problems/array-nesting/) - Medium

## Notes

- Works best when numbers are in range [1, n] or [0, n]
- Usually achieves O(n) time complexity with O(1) space
- Useful for finding duplicates and missing numbers

---

**Tags:** #dsa #cyclic-sort #leetcode

**Last Reviewed:** 2025-11-05