# K-way Merge

## Explanation

The K-way Merge pattern efficiently merges K sorted arrays or lists by using a min-heap to always select the smallest element. It maintains pointers to the current element in each array and adds the minimum to the result. This pattern is ideal for merging sorted data streams or finding the K smallest/largest elements across multiple sorted arrays.

## Algorithm (Word-based)

- Initialize min-heap with first element from each sorted array
- Track array index and element index with each heap entry
- While heap is not empty
- Extract minimum element and add to result
- Add next element from the same array to heap
- Continue until all elements processed

## Code Template

```python
import heapq

def k_way_merge(sorted_arrays):
    """
    Merge K sorted arrays into one sorted array
    """
    # Min-heap: (value, array_index, element_index)
    heap = []
    result = []
    
    # Initialize heap with first element from each array
    for i, arr in enumerate(sorted_arrays):
        if arr:  # Check if array is not empty
            heapq.heappush(heap, (arr[0], i, 0))
    
    # Process elements from heap
    while heap:
        value, array_idx, element_idx = heapq.heappop(heap)
        result.append(value)
        
        # Add next element from the same array if exists
        if element_idx + 1 < len(sorted_arrays[array_idx]):
            next_val = sorted_arrays[array_idx][element_idx + 1]
            heapq.heappush(heap, (next_val, array_idx, element_idx + 1))
    
    return result

def k_smallest_in_sorted_matrix(matrix, k):
    """
    Find kth smallest element in sorted matrix (N×N)
    """
    heap = []
    n = len(matrix)
    
    # Add first element of each row (at most k rows)
    for i in range(min(k, n)):
        heapq.heappush(heap, (matrix[i][0], i, 0))
    
    # Extract k-1 smallest elements
    for _ in range(k - 1):
        value, row, col = heapq.heappop(heap)
        
        # Add next element from same row
        if col + 1 < n:
            heapq.heappush(heap, (matrix[row][col + 1], row, col + 1))
    
    # Return kth smallest
    return heap[0][0] if heap else None
```

## Practice Questions

1. [Merge K Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists/) - Hard
2. [Merge Sorted Array](https://leetcode.com/problems/merge-sorted-array/) - Easy
3. [Kth Smallest Element in a Sorted Matrix](https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/) - Medium
4. [Find K Pairs with Smallest Sums](https://leetcode.com/problems/find-k-pairs-with-smallest-sums/) - Medium
5. [Smallest Range Covering Elements from K Lists](https://leetcode.com/problems/smallest-range-covering-elements-from-k-lists/) - Hard
6. [Merge Intervals](https://leetcode.com/problems/merge-intervals/) - Medium
7. [Kth Largest Element in an Array](https://leetcode.com/problems/kth-largest-element-in-an-array/) - Medium
8. [Ugly Number II](https://leetcode.com/problems/ugly-number-ii/) - Medium
9. [Super Ugly Number](https://leetcode.com/problems/super-ugly-number/) - Medium
10. [Find K-th Smallest Pair Distance](https://leetcode.com/problems/find-k-th-smallest-pair-distance/) - Hard

## Notes

- Time complexity: O(N log K) where N is total elements and K is number of arrays
- Space complexity: O(K) for the heap
- Useful for merging sorted streams efficiently

---

**Tags:** #dsa #k-way-merge #heap #leetcode

**Last Reviewed:** 2025-11-05