# Modified Binary Search

## Explanation

Modified binary search extends the basic binary search concept to handle more complex scenarios like rotated arrays, finding peaks, or searching in 2D matrices. It maintains the O(log n) time complexity while adapting the comparison logic to work with non-trivial data patterns. This pattern is essential for advanced searching problems.

## Algorithm (Word-based)

- Define boundaries for search space
- Calculate mid point
- Apply modified condition based on problem constraints
- Update boundaries according to problem-specific logic
- Handle edge cases and special conditions
- Continue until target is found or space is exhausted

## Code Template

```python
def modified_binary_search(arr, target):
    """
    Template for modified binary search based on problem
    """
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        # Custom logic based on problem
        if condition_met(arr, mid, target):
            return mid  # Found solution
        elif need_to_search_left(arr, mid, target):
            right = mid - 1  # Search left half
        else:
            left = mid + 1   # Search right half
    
    return -1  # Solution not found
```

## Practice Questions

1. [Search in Rotated Sorted Array](https://leetcode.com/problems/search-in-rotated-sorted-array/) - Medium
2. [Find Minimum in Rotated Sorted Array](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/) - Medium
3. [Search in Rotated Sorted Array II](https://leetcode.com/problems/search-in-rotated-sorted-array-ii/) - Medium
4. [Find Peak Element](https://leetcode.com/problems/find-peak-element/) - Medium
5. [Find Minimum in Rotated Sorted Array II](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array-ii/) - Hard
6. [Search a 2D Matrix II](https://leetcode.com/problems/search-a-2d-matrix-ii/) - Medium
7. [Single Element in a Sorted Array](https://leetcode.com/problems/single-element-in-a-sorted-array/) - Medium
8. [Capacity To Ship Packages Within D Days](https://leetcode.com/problems/capacity-to-ship-packages-within-d-days/) - Medium
9. [Split Array Largest Sum](https://leetcode.com/problems/split-array-largest-sum/) - Hard
10. [Minimum Number of Days to Make m Bouquets](https://leetcode.com/problems/minimum-number-of-days-to-make-m-bouquets/) - Medium

## Notes

- Adapts binary search to non-standard scenarios
- Requires custom conditions based on problem
- Maintains O(log n) efficiency when applicable

---

**Tags:** #dsa #modified-binary-search #leetcode

**Last Reviewed:** 2025-11-05