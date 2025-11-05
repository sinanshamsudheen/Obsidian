# Sliding Window

## Explanation

The sliding window technique involves creating a window over a portion of data and sliding it through the data to efficiently solve problems. This pattern is used when you need to maintain a subset of data that satisfies certain conditions. It's particularly effective for array/string problems where you're looking for subarrays/substrings with specific properties.

## Algorithm (Word-based)

- Initialize two pointers (left and right) to represent the window boundaries
- Expand the right pointer to include new elements
- Contract the left pointer when the window condition is violated
- Track the result while maintaining the window constraints
- Continue until the right pointer reaches the end

## Code Template

```python
def sliding_window_template(arr, k):
    """
    Sliding window approach for array with window size k
    """
    left = 0
    result = []
    
    for right in range(len(arr)):
        # Process element at right pointer
        
        # Shrink window if condition is met
        while condition_violated():
            # Process element at left pointer
            left += 1
        
        # Update result based on current window
        result.append(current_window_value)
    
    return result
```

## Practice Questions

1. [Maximum Sum Subarray of Size K](https://leetcode.com/problems/maximum-sum-subarray-of-size-k/) - Easy
2. [Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/) - Medium
3. [Minimum Window Substring](https://leetcode.com/problems/minimum-window-substring/) - Hard
4. [Sliding Window Maximum](https://leetcode.com/problems/sliding-window-maximum/) - Hard
5. [Fruit Into Baskets](https://leetcode.com/problems/fruit-into-baskets/) - Medium
6. [Longest Repeating Character Replacement](https://leetcode.com/problems/longest-repeating-character-replacement/) - Medium
7. [Permutation in String](https://leetcode.com/problems/permutation-in-string/) - Medium
8. [Count Number of Nice Subarrays](https://leetcode.com/problems/count-number-of-nice-subarrays/) - Medium
9. [Max Consecutive Ones III](https://leetcode.com/problems/max-consecutive-ones-iii/) - Medium
10. [Grumpy Bookstore Owner](https://leetcode.com/problems/grumpy-bookstore-owner/) - Medium

## Notes

- Can be fixed size or variable size window
- Often reduces O(n²) complexity to O(n)
- Key insight is that elements in between left and right pointers are already processed

---

**Tags:** #dsa #sliding-window #leetcode

**Last Reviewed:** 2025-11-05