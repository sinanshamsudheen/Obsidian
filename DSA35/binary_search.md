# Binary Search

## Explanation

Binary search is an efficient algorithm for finding an element in a sorted array by repeatedly dividing the search space in half. It works by comparing the target with the middle element and eliminating half of the remaining elements. This pattern achieves O(log n) time complexity, making it much faster than linear search.

## Algorithm (Word-based)

- Initialize left pointer at start, right pointer at end
- While left <= right
- Calculate mid index as (left + right) // 2
- If mid element equals target, return index
- If target is smaller, search left half
- If target is larger, search right half
- Return -1 if not found

## Code Template

```python
def binary_search(arr, target):
    """
    Standard binary search in sorted array
    """
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = left + (right - left) // 2  # Prevents overflow
        
        if arr[mid] == target:
            return mid  # Found target
        elif arr[mid] < target:
            left = mid + 1  # Search right half
        else:
            right = mid - 1  # Search left half
    
    return -1  # Target not found
```

## Practice Questions

1. [Binary Search](https://leetcode.com/problems/binary-search/) - Easy
2. [Search Insert Position](https://leetcode.com/problems/search-insert-position/) - Easy
3. [First Bad Version](https://leetcode.com/problems/first-bad-version/) - Easy
4. [Find Peak Element](https://leetcode.com/problems/find-peak-element/) - Medium
5. [Search in Rotated Sorted Array](https://leetcode.com/problems/search-in-rotated-sorted-array/) - Medium
6. [Find First and Last Position of Element in Sorted Array](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/) - Medium
7. [Find Minimum in Rotated Sorted Array](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/) - Medium
8. [Sqrt(x)](https://leetcode.com/problems/sqrtx/) - Easy
9. [Koko Eating Bananas](https://leetcode.com/problems/koko-eating-bananas/) - Medium
10. [Search a 2D Matrix](https://leetcode.com/problems/search-a-2d-matrix/) - Medium

## Notes

- Always use left + (right - left) // 2 to prevent overflow
- Works only on sorted arrays
- Can be modified to find boundaries

---

**Tags:** #dsa #binary-search #leetcode

**Last Reviewed:** 2025-11-05