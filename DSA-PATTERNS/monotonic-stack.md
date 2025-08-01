## 🧠 Monotonic Stack

**Monotonic Stack** maintains elements in monotonic (increasing or decreasing) order, automatically removing elements that violate the order when new elements are added.

• Stack maintains either increasing or decreasing order from bottom to top
• When adding element, pop elements that violate monotonic property
• Useful for finding next/previous greater/smaller elements efficiently
• Time complexity: O(n) for processing n elements, each element pushed/popped at most once

**Related:** [[Stack]] [[Array]] [[Optimization]]

---

## 💡 Python Code Snippet

```python
def next_greater_element(nums):
    """
    Find next greater element for each element in array
    """
    result = [-1] * len(nums)
    stack = []  # Monotonic decreasing stack (indices)
    
    for i in range(len(nums)):
        # Pop elements smaller than current element
        while stack and nums[stack[-1]] < nums[i]:
            idx = stack.pop()
            result[idx] = nums[i]
        
        stack.append(i)
    
    return result

def next_greater_element_circular(nums):
    """
    Next greater element in circular array
    """
    n = len(nums)
    result = [-1] * n
    stack = []
    
    # Process array twice to handle circular nature
    for i in range(2 * n):
        current_idx = i % n
        
        # Pop elements smaller than current element
        while stack and nums[stack[-1]] < nums[current_idx]:
            idx = stack.pop()
            result[idx] = nums[current_idx]
        
        # Only add to stack in first iteration to avoid duplicates
        if i < n:
            stack.append(current_idx)
    
    return result

def daily_temperatures(temperatures):
    """
    Days until warmer temperature (Next Greater Element variation)
    """
    result = [0] * len(temperatures)
    stack = []  # Monotonic decreasing stack (indices)
    
    for i, temp in enumerate(temperatures):
        # Pop days with cooler temperatures
        while stack and temperatures[stack[-1]] < temp:
            prev_day = stack.pop()
            result[prev_day] = i - prev_day
        
        stack.append(i)
    
    return result

def largest_rectangle_histogram(heights):
    """
    Largest rectangle area in histogram using monotonic stack
    """
    stack = []  # Monotonic increasing stack (indices)
    max_area = 0
    
    for i, height in enumerate(heights):
        # Pop taller bars and calculate area
        while stack and heights[stack[-1]] > height:
            h = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, h * width)
        
        stack.append(i)
    
    # Process remaining bars in stack
    while stack:
        h = heights[stack.pop()]
        width = len(heights) if not stack else len(heights) - stack[-1] - 1
        max_area = max(max_area, h * width)
    
    return max_area

def sliding_window_maximum(nums, k):
    """
    Maximum in sliding window using monotonic deque
    """
    from collections import deque
    
    result = []
    dq = deque()  # Monotonic decreasing deque (indices)
    
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

def remove_k_digits(num, k):
    """
    Remove k digits to form smallest number using monotonic stack
    """
    stack = []
    
    for digit in num:
        # Remove larger digits to make number smaller
        while stack and stack[-1] > digit and k > 0:
            stack.pop()
            k -= 1
        
        stack.append(digit)
    
    # Remove remaining digits from end
    while k > 0:
        stack.pop()
        k -= 1
    
    # Build result, handle leading zeros
    result = ''.join(stack).lstrip('0')
    return result if result else '0'

def stock_spanner():
    """
    Stock price span using monotonic stack
    """
    class StockSpanner:
        def __init__(self):
            self.stack = []  # Monotonic decreasing stack (price, span)
        
        def next(self, price):
            span = 1
            
            # Pop prices less than or equal to current price
            while self.stack and self.stack[-1][0] <= price:
                _, prev_span = self.stack.pop()
                span += prev_span
            
            self.stack.append((price, span))
            return span
    
    return StockSpanner()

def previous_smaller_element(nums):
    """
    Find previous smaller element for each element
    """
    result = [-1] * len(nums)
    stack = []  # Monotonic increasing stack (values)
    
    for i, num in enumerate(nums):
        # Pop elements greater than or equal to current
        while stack and stack[-1] >= num:
            stack.pop()
        
        # Previous smaller element is top of stack
        if stack:
            result[i] = stack[-1]
        
        stack.append(num)
    
    return result

def sum_of_subarray_minimums(arr):
    """
    Sum of minimum of all subarrays using monotonic stack
    """
    MOD = 10**9 + 7
    n = len(arr)
    
    # Find previous smaller element indices
    prev_smaller = [-1] * n
    stack = []
    for i in range(n):
        while stack and arr[stack[-1]] >= arr[i]:
            stack.pop()
        if stack:
            prev_smaller[i] = stack[-1]
        stack.append(i)
    
    # Find next smaller element indices
    next_smaller = [n] * n
    stack = []
    for i in range(n - 1, -1, -1):
        while stack and arr[stack[-1]] > arr[i]:
            stack.pop()
        if stack:
            next_smaller[i] = stack[-1]
        stack.append(i)
    
    # Calculate contribution of each element
    result = 0
    for i in range(n):
        left_count = i - prev_smaller[i]
        right_count = next_smaller[i] - i
        contribution = (arr[i] * left_count * right_count) % MOD
        result = (result + contribution) % MOD
    
    return result

def maximal_rectangle(matrix):
    """
    Largest rectangle in binary matrix using monotonic stack
    """
    if not matrix or not matrix[0]:
        return 0
    
    rows, cols = len(matrix), len(matrix[0])
    heights = [0] * cols
    max_area = 0
    
    for row in matrix:
        # Update heights array
        for i in range(cols):
            if row[i] == '1':
                heights[i] += 1
            else:
                heights[i] = 0
        
        # Find largest rectangle in current histogram
        max_area = max(max_area, largest_rectangle_histogram(heights))
    
    return max_area

def car_fleet(target, position, speed):
    """
    Number of car fleets using monotonic stack concept
    """
    # Calculate time to reach target for each car
    cars = [(pos, (target - pos) / spd) for pos, spd in zip(position, speed)]
    cars.sort()  # Sort by position
    
    fleets = 0
    slowest_time = 0
    
    # Process from car closest to target
    for pos, time in reversed(cars):
        # If current car takes longer, it forms a new fleet
        if time > slowest_time:
            fleets += 1
            slowest_time = time
    
    return fleets

def shortest_unsorted_subarray(nums):
    """
    Find shortest subarray to sort using monotonic stack
    """
    n = len(nums)
    left_bound = n
    right_bound = -1
    
    # Use monotonic increasing stack to find left boundary
    stack = []
    for i in range(n):
        while stack and nums[stack[-1]] > nums[i]:
            left_bound = min(left_bound, stack.pop())
        stack.append(i)
    
    # Use monotonic decreasing stack to find right boundary
    stack = []
    for i in range(n - 1, -1, -1):
        while stack and nums[stack[-1]] < nums[i]:
            right_bound = max(right_bound, stack.pop())
        stack.append(i)
    
    return 0 if right_bound == -1 else right_bound - left_bound + 1

def create_maximum_number(nums1, nums2, k):
    """
    Create maximum number using monotonic stack approach
    """
    def max_array(nums, k):
        """Extract k digits to form maximum number"""
        stack = []
        to_remove = len(nums) - k
        
        for num in nums:
            while stack and stack[-1] < num and to_remove > 0:
                stack.pop()
                to_remove -= 1
            stack.append(num)
        
        return stack[:k]
    
    def merge(arr1, arr2):
        """Merge two arrays to form maximum number"""
        result = []
        i = j = 0
        
        while i < len(arr1) and j < len(arr2):
            if arr1[i:] > arr2[j:]:  # Lexicographically compare remaining parts
                result.append(arr1[i])
                i += 1
            else:
                result.append(arr2[j])
                j += 1
        
        result.extend(arr1[i:])
        result.extend(arr2[j:])
        return result
    
    max_result = []
    
    # Try all possible distributions of k elements
    for i in range(max(0, k - len(nums2)), min(k, len(nums1)) + 1):
        arr1 = max_array(nums1, i)
        arr2 = max_array(nums2, k - i)
        merged = merge(arr1, arr2)
        
        if merged > max_result:
            max_result = merged
    
    return max_result
```

---

## 🔗 LeetCode Practice Problems (10)

- **Daily Temperatures** – https://leetcode.com/problems/daily-temperatures/
- **Next Greater Element I** – https://leetcode.com/problems/next-greater-element-i/
- **Next Greater Element II** – https://leetcode.com/problems/next-greater-element-ii/
- **Largest Rectangle in Histogram** – https://leetcode.com/problems/largest-rectangle-in-histogram/
- **Sliding Window Maximum** – https://leetcode.com/problems/sliding-window-maximum/
- **Remove K Digits** – https://leetcode.com/problems/remove-k-digits/
- **Online Stock Span** – https://leetcode.com/problems/online-stock-span/
- **Sum of Subarray Minimums** – https://leetcode.com/problems/sum-of-subarray-minimums/
- **Maximal Rectangle** – https://leetcode.com/problems/maximal-rectangle/
- **Shortest Unsorted Continuous Subarray** – https://leetcode.com/problems/shortest-unsorted-continuous-subarray/

---

## 🧠 Flashcard (for spaced repetition)

What is the Monotonic Stack pattern? ::

• Maintains elements in monotonic order, removing elements that violate order when adding new ones
• Each element pushed/popped at most once, achieving O(n) total time complexity
• Used for next/previous greater/smaller elements, histogram problems, and optimization tasks

[[monotonic-stack]] 