# Kadane's Algorithm

## Explanation

Kadane's Algorithm efficiently finds the maximum sum of a contiguous subarray in an array of numbers. It works by keeping track of the maximum sum ending at each position and updating the global maximum. The algorithm runs in O(n) time, making it optimal for maximum subarray problems.

## Algorithm (Word-based)

- Initialize max_current and max_global to first element
- Iterate through array starting from second element
- At each position, decide whether to extend current subarray or start new one
- Update max_global if current sum is greater
- Continue until end of array

## Code Template

```python
def kadane_max_subarray(arr):
    """
    Find maximum sum of contiguous subarray using Kadane's algorithm
    """
    if not arr:
        return 0
    
    max_current = max_global = arr[0]
    
    for i in range(1, len(arr)):
        # Decide whether to include current element in existing subarray or start new
        max_current = max(arr[i], max_current + arr[i])
        
        # Update global maximum
        max_global = max(max_global, max_current)
    
    return max_global

def kadane_with_indices(arr):
    """
    Find maximum subarray with start and end indices
    """
    if not arr:
        return 0, -1, -1
    
    max_current = max_global = arr[0]
    start = end = 0
    temp_start = 0
    
    for i in range(1, len(arr)):
        if arr[i] > max_current + arr[i]:
            max_current = arr[i]
            temp_start = i
        else:
            max_current = max_current + arr[i]
        
        if max_current > max_global:
            max_global = max_current
            start = temp_start
            end = i
    
    return max_global, start, end

def max_subarray_circular(arr):
    """
    Maximum subarray sum in circular array
    """
    def kadane(nums):
        max_current = max_global = nums[0]
        for i in range(1, len(nums)):
            max_current = max(nums[i], max_current + nums[i])
            max_global = max(max_global, max_current)
        return max_global
    
    # Normal Kadane (non-circular)
    max_normal = kadane(arr)
    
    # If all numbers are negative, return normal result
    if max_normal < 0:
        return max_normal
    
    # Calculate array sum and invert signs for max circular sum
    arr_sum = sum(arr)
    inverted_arr = [-x for x in arr]
    max_inverted = kadane(inverted_arr)
    
    # Maximum circular sum = total_sum + max_subarray_of_inverted
    max_circular = arr_sum + max_inverted
    
    # Return maximum of normal and circular
    return max(max_normal, max_circular)
```

## Practice Questions

1. [Maximum Subarray](https://leetcode.com/problems/maximum-subarray/) - Easy
2. [Maximum Product Subarray](https://leetcode.com/problems/maximum-product-subarray/) - Medium
3. [Maximum Subarray Sum After One Deletion](https://leetcode.com/problems/maximum-subarray-sum-with-one-deletion/) - Medium
4. [Maximum Sum Circular Subarray](https://leetcode.com/problems/maximum-sum-circular-subarray/) - Medium
5. [Best Time to Buy and Sell Stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/) - Easy
6. [Best Time to Buy and Sell Stock II](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/) - Medium
7. [Best Time to Buy and Sell Stock III](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/) - Hard
8. [Best Time to Buy and Sell Stock IV](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iv/) - Hard
9. [Best Time to Buy and Sell Stock with Cooldown](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/) - Medium
10. [Best Time to Buy and Sell Stock with Transaction Fee](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/) - Medium

## Notes

- Time complexity: O(n), Space complexity: O(1)
- Can be extended to find indices of the maximum subarray
- Key insight: At each position, decide to extend or start fresh

---

**Tags:** #dsa #kadane-algorithm #subarray #leetcode

**Last Reviewed:** 2025-11-05