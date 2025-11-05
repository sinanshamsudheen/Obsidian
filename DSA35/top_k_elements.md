# Top K Elements (Heap)

## Explanation

The Top K Elements pattern uses heaps to efficiently find the K largest or smallest elements in a dataset. A min-heap is used for finding K largest elements (keeping only K elements in heap), while a max-heap is used for K smallest elements. This pattern is efficient for streaming data and when K << N.

## Algorithm (Word-based)

- Create appropriate heap (min for largest K, max for smallest K)
- For each element in dataset
- If heap size < K, add element
- If element is better than top of heap, replace top
- Return elements in heap as result

## Code Template

```python
import heapq

def top_k_elements(nums, k, largest=True):
    """
    Find top K elements using heap
    """
    if not nums or k <= 0:
        return []
    
    if largest:
        # Use min-heap to keep the K largest elements
        heap = []
        for num in nums:
            if len(heap) < k:
                heapq.heappush(heap, num)
            elif num > heap[0]:  # If current number is larger than smallest in heap
                heapq.heapreplace(heap, num)  # Remove smallest, add current
    else:
        # Use max-heap to keep the K smallest elements (negate values)
        heap = []
        for num in nums:
            if len(heap) < k:
                heapq.heappush(heap, -num)  # Negate to simulate max-heap
            elif -num > heap[0]:  # If current number is smaller than largest in heap
                heapq.heapreplace(heap, -num)
        
        # Convert back to positive values
        heap = [-x for x in heap]
    
    return heap

def top_k_with_frequency(nums, k):
    """
    Find top K elements by frequency
    """
    # Count frequencies
    freq_map = {}
    for num in nums:
        freq_map[num] = freq_map.get(num, 0) + 1
    
    # Use min-heap to keep K most frequent elements
    heap = []
    for num, freq in freq_map.items():
        if len(heap) < k:
            heapq.heappush(heap, (freq, num))
        elif freq > heap[0][0]:  # If current freq is higher than smallest in heap
            heapq.heapreplace(heap, (freq, num))
    
    return [num for freq, num in heap]
```

## Practice Questions

1. [Top K Frequent Elements](https://leetcode.com/problems/top-k-frequent-elements/) - Medium
2. [Kth Largest Element in an Array](https://leetcode.com/problems/kth-largest-element-in-an-array/) - Medium
3. [Top K Frequent Words](https://leetcode.com/problems/top-k-frequent-words/) - Medium
4. [K Closest Points to Origin](https://leetcode.com/problems/k-closest-points-to-origin/) - Medium
5. [Sort Characters By Frequency](https://leetcode.com/problems/sort-characters-by-frequency/) - Medium
6. [Reorganize String](https://leetcode.com/problems/reorganize-string/) - Medium
7. [Find K Pairs with Smallest Sums](https://leetcode.com/problems/find-k-pairs-with-smallest-sums/) - Medium
8. [Sliding Window Maximum](https://leetcode.com/problems/sliding-window-maximum/) - Hard
9. [Merge k Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists/) - Medium
10. [Find K-th Smallest Pair Distance](https://leetcode.com/problems/find-k-th-smallest-pair-distance/) - Hard

## Notes

- Min-heap for largest K elements, max-heap for smallest K elements
- Time complexity: O(N log K) where N is number of elements
- Space complexity: O(K) for the heap

---

**Tags:** #dsa #top-k-elements #heap #leetcode

**Last Reviewed:** 2025-11-05