# Two Pointers

## Explanation

The two pointers technique involves using two pointers to iterate through data structures, typically arrays or linked lists. One pointer may move from the beginning while another moves from the end, or one moves faster than the other. This pattern is efficient for searching pairs, removing duplicates, or comparing elements in sorted arrays.

## Algorithm (Word-based)

- Initialize two pointers (start and end, or slow and fast)
- Move pointers based on comparison or condition
- Process elements at pointer positions
- Stop when pointers meet or condition is satisfied
- Return result based on pointer positions or values

## Code Template

```python
def two_pointers_template(arr):
    """
    Two pointers approach for sorted array problems
    """
    left = 0
    right = len(arr) - 1
    
    while left < right:
        # Process elements at left and right pointers
        current_sum = arr[left] + arr[right]
        
        if condition_met(current_sum):
            # Found solution
            return result
        elif current_sum < target:
            left += 1  # Move left pointer forward
        else:
            right -= 1  # Move right pointer backward
    
    return default_result
```

## Practice Questions

1. [Two Sum II - Input Array Is Sorted](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/) - Easy
2. [Remove Duplicates from Sorted Array](https://leetcode.com/problems/remove-duplicates-from-sorted-array/) - Easy
3. [Squares of a Sorted Array](https://leetcode.com/problems/squares-of-a-sorted-array/) - Easy
4. [3Sum](https://leetcode.com/problems/3sum/) - Medium
5. [4Sum](https://leetcode.com/problems/4sum/) - Medium
6. [Container With Most Water](https://leetcode.com/problems/container-with-most-water/) - Medium
7. [Trapping Rain Water](https://leetcode.com/problems/trapping-rain-water/) - Hard
8. [Sort Colors](https://leetcode.com/problems/sort-colors/) - Medium
9. [Valid Palindrome](https://leetcode.com/problems/valid-palindrome/) - Easy
10. [Merge Sorted Array](https://leetcode.com/problems/merge-sorted-array/) - Easy

## Notes

- Often used with sorted arrays for efficient search
- Can be used for partitioning problems
- Time complexity typically O(n) with O(1) space

---

**Tags:** #dsa #two-pointers #leetcode

**Last Reviewed:** 2025-11-05