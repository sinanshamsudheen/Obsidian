## 🧠 K-way Merge

**K-way Merge** efficiently merges k sorted arrays or lists using a min-heap to always pick the smallest available element.

• Uses a min-heap to keep track of the smallest element from each of the k arrays
• Maintains k pointers, one for each array, advancing the pointer after picking an element
• Time complexity: O(n log k) where n is total number of elements
• Space complexity: O(k) for the heap

**Related:** [[Heap]] [[Array]] [[Merge Sort]]

---

## 💡 Python Code Snippet

```python
import heapq

def merge_k_sorted_arrays(arrays):
    """
    Merge k sorted arrays into one sorted array
    """
    if not arrays:
        return []
    
    heap = []
    result = []
    
    # Initialize heap with first element from each array
    for i, array in enumerate(arrays):
        if array:  # Check if array is not empty
            heapq.heappush(heap, (array[0], i, 0))  # (value, array_index, element_index)
    
    while heap:
        value, array_idx, element_idx = heapq.heappop(heap)
        result.append(value)
        
        # Add next element from the same array
        if element_idx + 1 < len(arrays[array_idx]):
            next_value = arrays[array_idx][element_idx + 1]
            heapq.heappush(heap, (next_value, array_idx, element_idx + 1))
    
    return result

def merge_k_sorted_lists(lists):
    """
    Merge k sorted linked lists
    """
    class ListNode:
        def __init__(self, val=0, next=None):
            self.val = val
            self.next = next
    
    if not lists:
        return None
    
    heap = []
    
    # Add first node from each list
    for i, head in enumerate(lists):
        if head:
            heapq.heappush(heap, (head.val, i, head))
    
    dummy = ListNode()
    current = dummy
    
    while heap:
        val, list_idx, node = heapq.heappop(heap)
        
        # Add node to result
        current.next = node
        current = current.next
        
        # Add next node from same list
        if node.next:
            heapq.heappush(heap, (node.next.val, list_idx, node.next))
    
    return dummy.next

def kth_smallest_in_sorted_matrix(matrix, k):
    """
    Find kth smallest element in row and column sorted matrix
    """
    if not matrix or not matrix[0]:
        return 0
    
    n = len(matrix)
    heap = []
    
    # Add first element from each row
    for i in range(min(n, k)):  # Only need to consider first k rows
        heapq.heappush(heap, (matrix[i][0], i, 0))
    
    for _ in range(k):
        value, row, col = heapq.heappop(heap)
        
        # If this is the kth pop, return the value
        if _ == k - 1:
            return value
        
        # Add next element from same row
        if col + 1 < n:
            heapq.heappush(heap, (matrix[row][col + 1], row, col + 1))
    
    return -1

def smallest_range_covering_k_lists(lists):
    """
    Find smallest range that contains at least one element from each list
    """
    heap = []
    max_val = float('-inf')
    
    # Initialize heap with first element from each list
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (lst[0], i, 0))
            max_val = max(max_val, lst[0])
    
    min_range = float('inf')
    result_range = [0, 0]
    
    while len(heap) == len(lists):  # All lists must be represented
        min_val, list_idx, element_idx = heapq.heappop(heap)
        
        # Update range if current is smaller
        if max_val - min_val < min_range:
            min_range = max_val - min_val
            result_range = [min_val, max_val]
        
        # Add next element from same list
        if element_idx + 1 < len(lists[list_idx]):
            next_val = lists[list_idx][element_idx + 1]
            heapq.heappush(heap, (next_val, list_idx, element_idx + 1))
            max_val = max(max_val, next_val)
    
    return result_range

def merge_sorted_streams():
    """
    Merge multiple sorted streams (infinite sequences)
    """
    class StreamMerger:
        def __init__(self, streams):
            self.heap = []
            self.streams = streams
            
            # Initialize with first element from each stream
            for i, stream in enumerate(streams):
                try:
                    value = next(stream)
                    heapq.heappush(self.heap, (value, i))
                except StopIteration:
                    pass
        
        def get_next(self):
            if not self.heap:
                raise StopIteration
            
            value, stream_idx = heapq.heappop(self.heap)
            
            # Try to get next value from same stream
            try:
                next_value = next(self.streams[stream_idx])
                heapq.heappush(self.heap, (next_value, stream_idx))
            except StopIteration:
                pass
            
            return value
    
    return StreamMerger

def k_pairs_smallest_sum(nums1, nums2, k):
    """
    Find k pairs with smallest sums from two arrays
    """
    if not nums1 or not nums2:
        return []
    
    heap = []
    result = []
    
    # Initialize heap with pairs (nums1[i], nums2[0])
    for i in range(min(len(nums1), k)):
        heapq.heappush(heap, (nums1[i] + nums2[0], i, 0))
    
    while heap and len(result) < k:
        sum_val, i, j = heapq.heappop(heap)
        result.append([nums1[i], nums2[j]])
        
        # Add next pair from same row
        if j + 1 < len(nums2):
            heapq.heappush(heap, (nums1[i] + nums2[j + 1], i, j + 1))
    
    return result

def merge_intervals_from_k_lists(interval_lists):
    """
    Merge overlapping intervals from k lists
    """
    if not interval_lists:
        return []
    
    # Collect all intervals
    all_intervals = []
    for intervals in interval_lists:
        all_intervals.extend(intervals)
    
    if not all_intervals:
        return []
    
    # Sort by start time
    all_intervals.sort(key=lambda x: x[0])
    
    merged = [all_intervals[0]]
    
    for current in all_intervals[1:]:
        last = merged[-1]
        
        if current[0] <= last[1]:  # Overlapping
            merged[-1] = [last[0], max(last[1], current[1])]
        else:
            merged.append(current)
    
    return merged

def kth_smallest_in_multiplication_table(m, n, k):
    """
    Find kth smallest element in m x n multiplication table
    """
    def count_less_equal(x):
        count = 0
        for i in range(1, m + 1):
            # How many numbers in row i are <= x
            count += min(x // i, n)
        return count
    
    left, right = 1, m * n
    
    while left < right:
        mid = left + (right - left) // 2
        
        if count_less_equal(mid) < k:
            left = mid + 1
        else:
            right = mid
    
    return left

def median_of_k_sorted_arrays(arrays):
    """
    Find median of k sorted arrays
    """
    # Merge all arrays
    merged = merge_k_sorted_arrays(arrays)
    
    if not merged:
        return 0
    
    n = len(merged)
    if n % 2 == 1:
        return merged[n // 2]
    else:
        return (merged[n // 2 - 1] + merged[n // 2]) / 2
```

---

## 🔗 LeetCode Practice Problems (10)

- **Merge k Sorted Lists** – https://leetcode.com/problems/merge-k-sorted-lists/
- **Kth Smallest Element in a Sorted Matrix** – https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/
- **Smallest Range Covering Elements from K Lists** – https://leetcode.com/problems/smallest-range-covering-elements-from-k-lists/
- **Find K Pairs with Smallest Sums** – https://leetcode.com/problems/find-k-pairs-with-smallest-sums/
- **Merge Sorted Array** – https://leetcode.com/problems/merge-sorted-array/
- **Kth Smallest Number in Multiplication Table** – https://leetcode.com/problems/kth-smallest-number-in-multiplication-table/
- **Ugly Number II** – https://leetcode.com/problems/ugly-number-ii/
- **Super Ugly Number** – https://leetcode.com/problems/super-ugly-number/
- **Merge Two Sorted Lists** – https://leetcode.com/problems/merge-two-sorted-lists/
- **Median of Two Sorted Arrays** – https://leetcode.com/problems/median-of-two-sorted-arrays/

---

## 🧠 Flashcard (for spaced repetition)

What is the K-way Merge pattern? :: • Efficiently merges k sorted arrays/lists using min-heap to pick smallest available element • Maintains k pointers advancing after picking elements, with O(n log k) time complexity • Used for merging multiple sorted sequences, finding kth smallest, and range problems [[k-way-merge]] 