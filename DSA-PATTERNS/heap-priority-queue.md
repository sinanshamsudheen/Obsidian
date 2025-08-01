## 🧠 Heap / Priority Queue

**Heap / Priority Queue** is a data structure that maintains elements in a partially ordered way, allowing efficient access to the minimum or maximum element.

• Two types: Min-heap (smallest element at root) and Max-heap (largest element at root)
• Key operations: insert O(log n), extract-min/max O(log n), peek O(1)
• Useful for problems involving "top K", "Kth largest/smallest", and scheduling
• Python's `heapq` module provides min-heap implementation

**Related:** [[Heap]] [[Tree]] [[Priority Queue]]

---

## 💡 Python Code Snippet

```python
import heapq
from collections import defaultdict

def kth_largest_element(nums, k):
    """
    Find kth largest element using min-heap of size k
    """
    heap = []
    
    for num in nums:
        heapq.heappush(heap, num)
        
        # Keep only k largest elements
        if len(heap) > k:
            heapq.heappop(heap)
    
    return heap[0]  # Root of min-heap is kth largest

def k_closest_points(points, k):
    """
    Find k closest points to origin using max-heap
    """
    heap = []
    
    for point in points:
        x, y = point
        distance = x*x + y*y
        
        # Use negative distance for max-heap behavior
        heapq.heappush(heap, (-distance, point))
        
        if len(heap) > k:
            heapq.heappop(heap)
    
    return [point for _, point in heap]

def merge_k_sorted_lists(lists):
    """
    Merge k sorted linked lists using min-heap
    """
    class ListNode:
        def __init__(self, val=0, next=None):
            self.val = val
            self.next = next
    
    heap = []
    
    # Add first node of each list to heap
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

def top_k_frequent_elements(nums, k):
    """
    Find k most frequent elements using min-heap
    """
    # Count frequency
    freq_map = defaultdict(int)
    for num in nums:
        freq_map[num] += 1
    
    heap = []
    
    for num, freq in freq_map.items():
        heapq.heappush(heap, (freq, num))
        
        if len(heap) > k:
            heapq.heappop(heap)
    
    return [num for freq, num in heap]

def find_median_data_stream():
    """
    Find median in data stream using two heaps
    """
    class MedianFinder:
        def __init__(self):
            self.max_heap = []  # Left half (negated for max-heap)
            self.min_heap = []  # Right half
        
        def addNum(self, num):
            # Add to max_heap first
            heapq.heappush(self.max_heap, -num)
            
            # Move largest from max_heap to min_heap
            heapq.heappush(self.min_heap, -heapq.heappop(self.max_heap))
            
            # Balance heaps
            if len(self.min_heap) > len(self.max_heap):
                heapq.heappush(self.max_heap, -heapq.heappop(self.min_heap))
        
        def findMedian(self):
            if len(self.max_heap) > len(self.min_heap):
                return -self.max_heap[0]
            else:
                return (-self.max_heap[0] + self.min_heap[0]) / 2
    
    return MedianFinder()

def sliding_window_maximum(nums, k):
    """
    Find maximum in sliding window using deque (alternative to heap)
    """
    from collections import deque
    
    result = []
    dq = deque()  # Store indices
    
    for i in range(len(nums)):
        # Remove elements outside window
        while dq and dq[0] <= i - k:
            dq.popleft()
        
        # Remove smaller elements from back
        while dq and nums[dq[-1]] <= nums[i]:
            dq.pop()
        
        dq.append(i)
        
        # Add to result if window is complete
        if i >= k - 1:
            result.append(nums[dq[0]])
    
    return result

def task_scheduler(tasks, n):
    """
    Task scheduling with cooldown using heap
    """
    from collections import Counter
    
    # Count task frequencies
    task_counts = Counter(tasks)
    heap = [-count for count in task_counts.values()]
    heapq.heapify(heap)
    
    time = 0
    
    while heap:
        temp = []
        
        # Execute tasks for one cycle (n+1 slots)
        for _ in range(n + 1):
            if heap:
                count = heapq.heappop(heap)
                if count < -1:  # Still has remaining executions
                    temp.append(count + 1)
            time += 1
            
            # If no more tasks, break early
            if not heap and not temp:
                break
        
        # Add remaining tasks back to heap
        for count in temp:
            heapq.heappush(heap, count)
    
    return time

class KthLargest:
    """
    Design class to find kth largest element in stream
    """
    def __init__(self, k, nums):
        self.k = k
        self.heap = nums
        heapq.heapify(self.heap)
        
        # Keep only k largest elements
        while len(self.heap) > k:
            heapq.heappop(self.heap)
    
    def add(self, val):
        heapq.heappush(self.heap, val)
        
        if len(self.heap) > self.k:
            heapq.heappop(self.heap)
        
        return self.heap[0]

def reorganize_string(s):
    """
    Reorganize string so no two adjacent characters are same
    """
    from collections import Counter
    
    # Count character frequencies
    char_count = Counter(s)
    heap = [(-count, char) for char, count in char_count.items()]
    heapq.heapify(heap)
    
    result = []
    prev_count, prev_char = 0, ''
    
    while heap:
        # Get most frequent character
        count, char = heapq.heappop(heap)
        result.append(char)
        
        # Add previous character back if it still has count
        if prev_count < 0:
            heapq.heappush(heap, (prev_count, prev_char))
        
        # Update previous character
        prev_count, prev_char = count + 1, char
    
    return ''.join(result) if len(result) == len(s) else ""
```

---

## 🔗 LeetCode Practice Problems (10)

- **Kth Largest Element in an Array** – https://leetcode.com/problems/kth-largest-element-in-an-array/
- **Top K Frequent Elements** – https://leetcode.com/problems/top-k-frequent-elements/
- **Merge k Sorted Lists** – https://leetcode.com/problems/merge-k-sorted-lists/
- **Find Median from Data Stream** – https://leetcode.com/problems/find-median-from-data-stream/
- **K Closest Points to Origin** – https://leetcode.com/problems/k-closest-points-to-origin/
- **Task Scheduler** – https://leetcode.com/problems/task-scheduler/
- **Kth Largest Element in a Stream** – https://leetcode.com/problems/kth-largest-element-in-a-stream/
- **Reorganize String** – https://leetcode.com/problems/reorganize-string/
- **Sliding Window Maximum** – https://leetcode.com/problems/sliding-window-maximum/
- **Ugly Number II** – https://leetcode.com/problems/ugly-number-ii/

---

## 🧠 Flashcard (for spaced repetition)

What is the Heap / Priority Queue pattern? ::

• Data structure maintaining partial order with efficient min/max access (insert/extract O(log n), peek O(1))
• Two types: min-heap (smallest at root) and max-heap (largest at root)
• Used for "top K", "Kth largest/smallest", scheduling, and streaming data problems

[[heap-priority-queue]] 