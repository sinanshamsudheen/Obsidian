## 🧠 Prefix Sum / Difference Arrays

**Prefix Sum / Difference Arrays** are preprocessing techniques that enable efficient range queries and updates by precomputing cumulative values or changes.

• Prefix Sum: Precompute cumulative sums for O(1) range sum queries
• Difference Array: Enable O(1) range updates by storing differences between consecutive elements
• Transforms O(n) range operations into O(1) operations with O(n) preprocessing
• Essential for subarray problems, range queries, and computational geometry

**Related:** [[Array]] [[Range Queries]] [[Preprocessing]] [[Optimization]]

---

## 💡 Python Code Snippet

```python
class PrefixSum:
    """
    Prefix sum for efficient range sum queries
    """
    def __init__(self, nums):
        self.prefix = [0] * (len(nums) + 1)
        for i in range(len(nums)):
            self.prefix[i + 1] = self.prefix[i] + nums[i]
    
    def range_sum(self, left, right):
        """Sum of elements from index left to right (inclusive)"""
        return self.prefix[right + 1] - self.prefix[left]
    
    def subarray_sum_equals_k(self, nums, k):
        """Count subarrays with sum equal to k"""
        prefix_sum = 0
        count = 0
        sum_count = {0: 1}  # prefix_sum -> frequency
        
        for num in nums:
            prefix_sum += num
            
            # Check if there's a prefix sum that when subtracted gives k
            if prefix_sum - k in sum_count:
                count += sum_count[prefix_sum - k]
            
            # Update frequency of current prefix sum
            sum_count[prefix_sum] = sum_count.get(prefix_sum, 0) + 1
        
        return count

def prefix_sum_2d(matrix):
    """
    2D prefix sum for rectangle sum queries
    """
    if not matrix or not matrix[0]:
        return []
    
    rows, cols = len(matrix), len(matrix[0])
    prefix = [[0] * (cols + 1) for _ in range(rows + 1)]
    
    # Build 2D prefix sum
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            prefix[i][j] = (matrix[i-1][j-1] + 
                           prefix[i-1][j] + 
                           prefix[i][j-1] - 
                           prefix[i-1][j-1])
    
    def range_sum_2d(r1, c1, r2, c2):
        """Sum of rectangle from (r1,c1) to (r2,c2) inclusive"""
        return (prefix[r2+1][c2+1] - 
                prefix[r1][c2+1] - 
                prefix[r2+1][c1] + 
                prefix[r1][c1])
    
    return range_sum_2d

class DifferenceArray:
    """
    Difference array for efficient range updates
    """
    def __init__(self, nums):
        self.diff = [0] * (len(nums) + 1)
        if nums:
            self.diff[0] = nums[0]
            for i in range(1, len(nums)):
                self.diff[i] = nums[i] - nums[i-1]
    
    def range_update(self, left, right, val):
        """Add val to all elements from left to right (inclusive)"""
        self.diff[left] += val
        self.diff[right + 1] -= val
    
    def get_array(self):
        """Reconstruct original array after updates"""
        result = []
        current = 0
        for i in range(len(self.diff) - 1):
            current += self.diff[i]
            result.append(current)
        return result

def max_subarray_sum_kadane(nums):
    """
    Maximum subarray sum using prefix sum concept
    """
    max_sum = current_sum = nums[0]
    
    for i in range(1, len(nums)):
        current_sum = max(nums[i], current_sum + nums[i])
        max_sum = max(max_sum, current_sum)
    
    return max_sum

def continuous_subarray_sum(nums, k):
    """
    Check if array has continuous subarray with sum multiple of k
    """
    remainder_map = {0: -1}  # remainder -> earliest index
    prefix_sum = 0
    
    for i, num in enumerate(nums):
        prefix_sum += num
        remainder = prefix_sum % k
        
        if remainder in remainder_map:
            # Subarray length must be at least 2
            if i - remainder_map[remainder] > 1:
                return True
        else:
            remainder_map[remainder] = i
    
    return False

def product_except_self(nums):
    """
    Product of array except self using prefix/suffix products
    """
    n = len(nums)
    result = [1] * n
    
    # Forward pass: prefix products
    for i in range(1, n):
        result[i] = result[i-1] * nums[i-1]
    
    # Backward pass: suffix products
    suffix = 1
    for i in range(n-1, -1, -1):
        result[i] *= suffix
        suffix *= nums[i]
    
    return result

def range_sum_query_mutable():
    """
    Range sum query with updates using difference array concept
    """
    class NumArray:
        def __init__(self, nums):
            self.nums = nums[:]
            self.prefix = [0] * (len(nums) + 1)
            for i in range(len(nums)):
                self.prefix[i + 1] = self.prefix[i] + nums[i]
        
        def update(self, index, val):
            # Update the array and rebuild prefix
            self.nums[index] = val
            for i in range(index + 1, len(self.prefix)):
                self.prefix[i] = self.prefix[i-1] + self.nums[i-1]
        
        def sumRange(self, left, right):
            return self.prefix[right + 1] - self.prefix[left]
    
    return NumArray

def minimum_size_subarray_sum(target, nums):
    """
    Minimum size subarray with sum >= target (sliding window + prefix)
    """
    min_length = float('inf')
    left = 0
    current_sum = 0
    
    for right in range(len(nums)):
        current_sum += nums[right]
        
        # Shrink window while sum >= target
        while current_sum >= target:
            min_length = min(min_length, right - left + 1)
            current_sum -= nums[left]
            left += 1
    
    return min_length if min_length != float('inf') else 0

def corporate_flight_bookings(bookings, n):
    """
    Corporate flight bookings using difference array
    """
    diff = [0] * (n + 1)
    
    # Apply range updates using difference array
    for first, last, seats in bookings:
        diff[first - 1] += seats  # Convert to 0-indexed
        diff[last] -= seats
    
    # Reconstruct final array
    result = []
    current = 0
    for i in range(n):
        current += diff[i]
        result.append(current)
    
    return result

def car_pooling_capacity(trips, capacity):
    """
    Car pooling problem using difference array
    """
    # Find maximum location
    max_location = max(trip[2] for trip in trips)
    diff = [0] * (max_location + 1)
    
    # Apply passenger changes at pickup/dropoff points
    for passengers, start, end in trips:
        diff[start] += passengers
        diff[end] -= passengers
    
    # Check if capacity is exceeded at any point
    current_passengers = 0
    for change in diff:
        current_passengers += change
        if current_passengers > capacity:
            return False
    
    return True

def running_sum_array(nums):
    """
    Simple running sum (prefix sum) of array
    """
    for i in range(1, len(nums)):
        nums[i] += nums[i-1]
    return nums

def find_pivot_index(nums):
    """
    Find pivot index where left sum equals right sum
    """
    total_sum = sum(nums)
    left_sum = 0
    
    for i, num in enumerate(nums):
        right_sum = total_sum - left_sum - num
        
        if left_sum == right_sum:
            return i
        
        left_sum += num
    
    return -1

def maximum_score_after_operations(nums, k):
    """
    Maximum score after k operations using prefix sum
    """
    n = len(nums)
    
    # Calculate prefix sum
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + nums[i]
    
    max_score = 0
    
    # Try all possible subarrays of length <= k
    for length in range(1, min(k, n) + 1):
        for start in range(n - length + 1):
            subarray_sum = prefix[start + length] - prefix[start]
            max_score = max(max_score, subarray_sum)
    
    return max_score

def matrix_block_sum(mat, k):
    """
    Matrix block sum using 2D prefix sum
    """
    m, n = len(mat), len(mat[0])
    
    # Build 2D prefix sum
    prefix = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            prefix[i][j] = (mat[i-1][j-1] + 
                           prefix[i-1][j] + 
                           prefix[i][j-1] - 
                           prefix[i-1][j-1])
    
    result = [[0] * n for _ in range(m)]
    
    for i in range(m):
        for j in range(n):
            # Calculate block boundaries
            r1 = max(0, i - k)
            c1 = max(0, j - k)
            r2 = min(m - 1, i + k)
            c2 = min(n - 1, j + k)
            
            # Calculate sum using prefix sum
            result[i][j] = (prefix[r2+1][c2+1] - 
                           prefix[r1][c2+1] - 
                           prefix[r2+1][c1] + 
                           prefix[r1][c1])
    
    return result

def count_number_of_nice_subarrays(nums, k):
    """
    Count subarrays with exactly k odd numbers using prefix sum
    """
    # Convert to binary: odd -> 1, even -> 0
    for i in range(len(nums)):
        nums[i] = nums[i] % 2
    
    # Now find subarrays with sum exactly k
    prefix_sum = 0
    count = 0
    sum_count = {0: 1}
    
    for num in nums:
        prefix_sum += num
        
        if prefix_sum - k in sum_count:
            count += sum_count[prefix_sum - k]
        
        sum_count[prefix_sum] = sum_count.get(prefix_sum, 0) + 1
    
    return count

def maximum_sum_circular_subarray(nums):
    """
    Maximum sum circular subarray using prefix sum
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
    
    # Case 1: Normal maximum subarray
    max_kadane = kadane_max(nums)
    
    # Case 2: Circular maximum = total - minimum subarray
    total_sum = sum(nums)
    min_kadane = kadane_min(nums)
    max_circular = total_sum - min_kadane
    
    # Handle edge case where all elements are negative
    if max_circular == 0:
        return max_kadane
    
    return max(max_kadane, max_circular)
```

---

## 🔗 LeetCode Practice Problems (10)

- **Range Sum Query - Immutable** – https://leetcode.com/problems/range-sum-query-immutable/
- **Subarray Sum Equals K** – https://leetcode.com/problems/subarray-sum-equals-k/
- **Corporate Flight Bookings** – https://leetcode.com/problems/corporate-flight-bookings/
- **Car Pooling** – https://leetcode.com/problems/car-pooling/
- **Product of Array Except Self** – https://leetcode.com/problems/product-of-array-except-self/
- **Find Pivot Index** – https://leetcode.com/problems/find-pivot-index/
- **Matrix Block Sum** – https://leetcode.com/problems/matrix-block-sum/
- **Continuous Subarray Sum** – https://leetcode.com/problems/continuous-subarray-sum/
- **Count Number of Nice Subarrays** – https://leetcode.com/problems/count-number-of-nice-subarrays/
- **Maximum Sum Circular Subarray** – https://leetcode.com/problems/maximum-sum-circular-subarray/

---

## 🧠 Flashcard (for spaced repetition)

What is the Prefix Sum / Difference Arrays pattern? ::

• Prefix Sum: Precomputes cumulative sums for O(1) range sum queries after O(n) preprocessing
• Difference Array: Enables O(1) range updates by storing differences between consecutive elements  
• Applied to subarray problems, range queries, and problems requiring efficient range operations

[[prefix-sum-difference-arrays]]