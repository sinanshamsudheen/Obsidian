## 🧠 Kadane's Algorithm

**Kadane's Algorithm** finds the maximum sum of a contiguous subarray within a one-dimensional array in linear time.

• Uses dynamic programming approach to track maximum subarray ending at each position
• Key insight: at each position, either extend previous subarray or start new subarray
• Time complexity: O(n), Space complexity: O(1)
• Can be extended to handle variations like maximum product, circular arrays, and 2D grids

**Related:** [[Dynamic Programming]] [[Array]] [[Optimization]]

---

## 💡 Python Code Snippet

```python
def max_subarray_sum(nums):
    """
    Classic Kadane's algorithm: find maximum sum of contiguous subarray
    """
    if not nums:
        return 0
    
    max_sum = current_sum = nums[0]
    
    for i in range(1, len(nums)):
        # Either extend current subarray or start new one
        current_sum = max(nums[i], current_sum + nums[i])
        max_sum = max(max_sum, current_sum)
    
    return max_sum

def max_subarray_with_indices(nums):
    """
    Kadane's algorithm that also returns start and end indices
    """
    if not nums:
        return 0, -1, -1
    
    max_sum = current_sum = nums[0]
    start = end = temp_start = 0
    
    for i in range(1, len(nums)):
        if current_sum < 0:
            # Start new subarray
            current_sum = nums[i]
            temp_start = i
        else:
            # Extend current subarray
            current_sum += nums[i]
        
        if current_sum > max_sum:
            max_sum = current_sum
            start = temp_start
            end = i
    
    return max_sum, start, end

def max_subarray_no_adjacent(nums):
    """
    Maximum sum with no adjacent elements (House Robber pattern)
    """
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    
    prev2 = nums[0]
    prev1 = max(nums[0], nums[1])
    
    for i in range(2, len(nums)):
        current = max(prev1, prev2 + nums[i])
        prev2 = prev1
        prev1 = current
    
    return prev1

def max_product_subarray(nums):
    """
    Maximum product of contiguous subarray (handling negative numbers)
    """
    if not nums:
        return 0
    
    max_product = min_product = result = nums[0]
    
    for i in range(1, len(nums)):
        # When multiplying by negative number, max and min swap
        if nums[i] < 0:
            max_product, min_product = min_product, max_product
        
        max_product = max(nums[i], max_product * nums[i])
        min_product = min(nums[i], min_product * nums[i])
        
        result = max(result, max_product)
    
    return result

def max_circular_subarray(nums):
    """
    Maximum sum of circular subarray (array wraps around)
    """
    def kadane_max(arr):
        max_sum = current_sum = arr[0]
        for num in arr[1:]:
            current_sum = max(num, current_sum + num)
            max_sum = max(max_sum, current_sum)
        return max_sum
    
    def kadane_min(arr):
        min_sum = current_sum = arr[0]
        for num in arr[1:]:
            current_sum = min(num, current_sum + num)
            min_sum = min(min_sum, current_sum)
        return min_sum
    
    # Case 1: Maximum subarray is non-circular
    max_kadane = kadane_max(nums)
    
    # Case 2: Maximum subarray is circular
    # Total sum - minimum subarray sum
    total_sum = sum(nums)
    min_kadane = kadane_min(nums)
    max_circular = total_sum - min_kadane
    
    # Edge case: all elements are negative
    if max_circular == 0:
        return max_kadane
    
    return max(max_kadane, max_circular)

def max_subarray_sum_k_elements(nums, k):
    """
    Maximum sum of subarray with exactly k elements
    """
    if len(nums) < k:
        return 0
    
    # Initial window sum
    window_sum = sum(nums[:k])
    max_sum = window_sum
    
    # Slide the window
    for i in range(k, len(nums)):
        window_sum = window_sum - nums[i - k] + nums[i]
        max_sum = max(max_sum, window_sum)
    
    return max_sum

def max_sum_two_non_overlapping_subarrays(nums, first_len, second_len):
    """
    Maximum sum of two non-overlapping subarrays of given lengths
    """
    n = len(nums)
    
    # Calculate sum of all possible subarrays of length first_len and second_len
    first_sums = []
    second_sums = []
    
    # Calculate sums for first_len subarrays
    window_sum = sum(nums[:first_len])
    first_sums.append(window_sum)
    for i in range(first_len, n):
        window_sum = window_sum - nums[i - first_len] + nums[i]
        first_sums.append(window_sum)
    
    # Calculate sums for second_len subarrays
    window_sum = sum(nums[:second_len])
    second_sums.append(window_sum)
    for i in range(second_len, n):
        window_sum = window_sum - nums[i - second_len] + nums[i]
        second_sums.append(window_sum)
    
    # Find maximum sum of two non-overlapping subarrays
    max_sum = 0
    
    # Case 1: first_len subarray comes before second_len subarray
    max_first = 0
    for i in range(len(first_sums)):
        if i >= first_len - 1:
            max_first = max(max_first, first_sums[i - first_len + 1])
        if i + second_len <= len(second_sums):
            max_sum = max(max_sum, max_first + second_sums[i + 1])
    
    # Case 2: second_len subarray comes before first_len subarray
    max_second = 0
    for i in range(len(second_sums)):
        if i >= second_len - 1:
            max_second = max(max_second, second_sums[i - second_len + 1])
        if i + first_len <= len(first_sums):
            max_sum = max(max_sum, max_second + first_sums[i + 1])
    
    return max_sum

def max_submatrix_sum(matrix):
    """
    Maximum sum rectangle in 2D matrix (Kadane's in 2D)
    """
    if not matrix or not matrix[0]:
        return 0
    
    rows, cols = len(matrix), len(matrix[0])
    max_sum = float('-inf')
    
    # Try all possible top and bottom row combinations
    for top in range(rows):
        temp = [0] * cols
        
        for bottom in range(top, rows):
            # Add current row to temp array
            for j in range(cols):
                temp[j] += matrix[bottom][j]
            
            # Apply Kadane's algorithm on temp array
            current_max = kadane_max_array(temp)
            max_sum = max(max_sum, current_max)
    
    return max_sum

def kadane_max_array(arr):
    """Helper function for Kadane's algorithm"""
    max_sum = current_sum = arr[0]
    for num in arr[1:]:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)
    return max_sum

def buy_sell_stock_one_transaction(prices):
    """
    Maximum profit from one buy-sell transaction (modified Kadane's)
    """
    if len(prices) < 2:
        return 0
    
    min_price = prices[0]
    max_profit = 0
    
    for price in prices[1:]:
        # Update minimum price seen so far
        min_price = min(min_price, price)
        
        # Update maximum profit
        max_profit = max(max_profit, price - min_price)
    
    return max_profit

def max_alternating_sum(nums):
    """
    Maximum alternating sum of subsequence
    """
    # even: max sum ending with even-indexed element (we sell)
    # odd: max sum ending with odd-indexed element (we buy)
    even = odd = 0
    
    for num in nums:
        new_even = max(even, odd + num)  # Sell at current price
        new_odd = max(odd, even - num)   # Buy at current price
        even, odd = new_even, new_odd
    
    return even  # Always end with selling

def max_subarray_min_size(nums, min_size):
    """
    Maximum sum of subarray with at least min_size elements
    """
    n = len(nums)
    if n < min_size:
        return 0
    
    # Calculate prefix sums for efficient range sum queries
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + nums[i]
    
    max_sum = float('-inf')
    
    # Try all possible subarrays of length >= min_size
    for i in range(n - min_size + 1):
        for j in range(i + min_size - 1, n):
            current_sum = prefix[j + 1] - prefix[i]
            max_sum = max(max_sum, current_sum)
    
    return max_sum

def longest_turbulent_subarray(arr):
    """
    Length of longest turbulent subarray (alternating comparisons)
    """
    if len(arr) <= 1:
        return len(arr)
    
    increasing = decreasing = 1
    max_length = 1
    
    for i in range(1, len(arr)):
        if arr[i] > arr[i - 1]:
            increasing = decreasing + 1
            decreasing = 1
        elif arr[i] < arr[i - 1]:
            decreasing = increasing + 1
            increasing = 1
        else:
            increasing = decreasing = 1
        
        max_length = max(max_length, max(increasing, decreasing))
    
    return max_length
```

---

## 🔗 LeetCode Practice Problems (10)

- **Maximum Subarray** – https://leetcode.com/problems/maximum-subarray/
- **Maximum Product Subarray** – https://leetcode.com/problems/maximum-product-subarray/
- **Maximum Sum Circular Subarray** – https://leetcode.com/problems/maximum-sum-circular-subarray/
- **Best Time to Buy and Sell Stock** – https://leetcode.com/problems/best-time-to-buy-and-sell-stock/
- **House Robber** – https://leetcode.com/problems/house-robber/
- **Maximum Sum of Two Non-Overlapping Subarrays** – https://leetcode.com/problems/maximum-sum-of-two-non-overlapping-subarrays/
- **Maximum Submatrix** – https://leetcode.com/problems/max-sum-of-rectangle-no-larger-than-k/
- **Maximum Alternating Subsequence Sum** – https://leetcode.com/problems/maximum-alternating-subsequence-sum/
- **Longest Turbulent Subarray** – https://leetcode.com/problems/longest-turbulent-subarray/
- **Maximum Average Subarray I** – https://leetcode.com/problems/maximum-average-subarray-i/

---

## 🧠 Flashcard (for spaced repetition)

What is the Kadane's Algorithm pattern? ::

• Finds maximum sum contiguous subarray in O(n) time using dynamic programming approach
• Key insight: at each position either extend previous subarray or start new one
• Extended to maximum product, circular arrays, 2D matrices, and stock trading problems

[[kadanes-algorithm]] 