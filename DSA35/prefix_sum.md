# Prefix Sum

## Explanation

The Prefix Sum pattern precomputes cumulative sums to answer range sum queries efficiently. By storing the sum from the beginning to each position, we can calculate any range sum in constant time. This pattern is essential for problems involving range queries or finding subarrays with specific properties.

## Algorithm (Word-based)

- Create prefix sum array where each element is sum of all previous elements
- Calculate prefix sum for each position as: prefix[i] = prefix[i-1] + arr[i]
- To find sum of range [i, j], use: prefix[j] - prefix[i-1]
- Handle edge case when i=0 separately

## Code Template

```python
class PrefixSum:
    def __init__(self, arr):
        """
        Initialize prefix sum array
        """
        self.prefix = [0]
        for num in arr:
            self.prefix.append(self.prefix[-1] + num)
    
    def range_sum(self, left, right):
        """
        Get sum of elements from index left to right (inclusive)
        """
        return self.prefix[right + 1] - self.prefix[left]

def find_subarray_sum_equals_k(nums, k):
    """
    Find if there exists a subarray with sum equal to k
    """
    prefix_sum = 0
    sum_map = {0: -1}  # Initialize with 0 to handle subarrays from start
    
    for i, num in enumerate(nums):
        prefix_sum += num
        
        # If (prefix_sum - k) exists in map, subarray exists
        if prefix_sum - k in sum_map:
            return True  # Found subarray from sum_map[prefix_sum-k]+1 to i
        
        # Store current prefix sum if not already present
        if prefix_sum not in sum_map:
            sum_map[prefix_sum] = i
    
    return False

def find_subarray_sum_equals_k_all(nums, k):
    """
    Find all subarrays with sum equal to k
    """
    result = []
    prefix_sum = 0
    sum_map = {0: [-1]}  # Store all indices for each prefix sum
    
    for i, num in enumerate(nums):
        prefix_sum += num
        
        # If (prefix_sum - k) exists in map, add all subarrays
        if prefix_sum - k in sum_map:
            for start_idx in sum_map[prefix_sum - k]:
                result.append(nums[start_idx + 1:i + 1])
        
        # Add current index to map
        if prefix_sum not in sum_map:
            sum_map[prefix_sum] = []
        sum_map[prefix_sum].append(i)
    
    return result
```

## Practice Questions

1. [Subarray Sum Equals K](https://leetcode.com/problems/subarray-sum-equals-k/) - Medium
2. [Range Sum Query - Immutable](https://leetcode.com/problems/range-sum-query-immutable/) - Easy
3. [Find Pivot Index](https://leetcode.com/problems/find-pivot-index/) - Easy
4. [Contiguous Array](https://leetcode.com/problems/contiguous-array/) - Medium
5. [Maximum Size Subarray Sum Equals k](https://leetcode.com/problems/maximum-size-subarray-sum-equals-k/) - Medium
6. [Continuous Subarray Sum](https://leetcode.com/problems/continuous-subarray-sum/) - Medium
7. [Maximum Points You Can Obtain from Cards](https://leetcode.com/problems/maximum-points-you-can-obtain-from-cards/) - Medium
8. [Minimum Operations to Reduce X to Zero](https://leetcode.com/problems/minimum-operations-to-reduce-x-to-zero/) - Medium
9. [Ways to Split Array Into Three Subarrays](https://leetcode.com/problems/ways-to-split-array-into-three-subarrays/) - Medium
10. [Number of Submatrices That Sum to Target](https://leetcode.com/problems/number-of-submatrices-that-sum-to-target/) - Hard

## Notes

- Standard prefix sum: O(1) range query after O(n) preprocessing
- Can be extended with hash maps for specific sum problems
- Useful for finding subarrays with particular properties

---

**Tags:** #dsa #prefix-sum #leetcode

**Last Reviewed:** 2025-11-05