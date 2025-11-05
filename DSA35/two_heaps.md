# Two Heaps

## Explanation

The Two Heaps pattern uses a max-heap and min-heap together to solve problems requiring quick access to both minimum and maximum values. Common applications include maintaining medians from a data stream, finding the middle value efficiently, and handling problems with two different sets of values that need to be balanced.

## Algorithm (Word-based)

- Use a max-heap to store smaller half of numbers
- Use a min-heap to store larger half of numbers
- Balance heaps to keep sizes differ by at most 1
- For median, return top of larger heap or average of both tops
- When adding number, decide which heap to add to based on value

## Code Template

```python
import heapq

class MedianFinder:
    """
    Find median from data stream using two heaps
    """
    def __init__(self):
        self.max_heap = []  # Store smaller half (negated for max-heap behavior)
        self.min_heap = []  # Store larger half
    
    def add_number(self, num):
        # Add to max_heap first (negated)
        heapq.heappush(self.max_heap, -num)
        
        # Move top of max_heap to min_heap
        heapq.heappush(self.min_heap, -heapq.heappop(self.max_heap))
        
        # Balance heaps if needed
        if len(self.min_heap) > len(self.max_heap):
            heapq.heappush(self.max_heap, -heapq.heappop(self.min_heap))
    
    def find_median(self):
        if len(self.max_heap) > len(self.min_heap):
            return -self.max_heap[0]  # Return max from smaller half
        else:
            return (-self.max_heap[0] + self.min_heap[0]) / 2  # Average of both tops

def sliding_window_median(nums, k):
    """
    Find median of sliding window using two heaps
    """
    result = []
    finder = MedianFinder()
    
    # Add first k elements
    for i in range(k):
        finder.add_number(nums[i])
    
    result.append(finder.find_median())
    
    # Slide window by removing leftmost element and adding next element
    for i in range(k, len(nums)):
        # Remove nums[i-k] and add nums[i]
        finder.remove_number(nums[i-k])
        finder.add_number(nums[i])
        result.append(finder.find_median())
    
    return result
```

## Practice Questions

1. [Find Median from Data Stream](https://leetcode.com/problems/find-median-from-data-stream/) - Hard
2. [Sliding Window Median](https://leetcode.com/problems/sliding-window-median/) - Hard
3. [IPO](https://leetcode.com/problems/ipo/) - Hard
4. [Maximum Subsequence Score](https://leetcode.com/problems/maximum-subsequence-score/) - Medium
5. [Total Cost to Hire K Workers](https://leetcode.com/problems/total-cost-to-hire-k-workers/) - Medium
6. [Minimize Deviation in Array](https://leetcode.com/problems/minimize-deviation-in-array/) - Hard
7. [Maximum Bags With Full Capacity of Rocks](https://leetcode.com/problems/maximum-bags-with-full-capacity-of-rocks/) - Medium
8. [Task Scheduler](https://leetcode.com/problems/task-scheduler/) - Medium
9. [Split Array into Consecutive Subsequences](https://leetcode.com/problems/split-array-into-consecutive-subsequences/) - Medium
10. [Maximum Performance of a Team](https://leetcode.com/problems/maximum-performance-of-a-team/) - Hard

## Notes

- Max-heap contains smaller half, min-heap contains larger half
- Balance heaps to keep sizes differ by at most 1
- Median is top of larger heap or average of both tops

---

**Tags:** #dsa #two-heaps #heap #leetcode

**Last Reviewed:** 2025-11-05