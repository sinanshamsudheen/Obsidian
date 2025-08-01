## 🧠 Modified Binary Search

**Modified Binary Search** adapts the binary search technique for problems beyond simple sorted array searches.

• Applies binary search to problems where we can eliminate half the search space based on some condition
• Works on rotated arrays, 2D matrices, or when searching for a condition rather than exact value
• Still maintains O(log n) time complexity by halving search space
• Key is identifying the condition that allows us to eliminate half the possibilities

**Related:** [[Array]] [[Binary Search]] [[Search Space]]

---

## 💡 Python Code Snippet

```python
def search_rotated_array(nums, target):
    """
    Search in rotated sorted array
    """
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            return mid
        
        # Check which half is sorted
        if nums[left] <= nums[mid]:  # Left half is sorted
            if nums[left] <= target < nums[mid]:
                right = mid - 1  # Target in left half
            else:
                left = mid + 1   # Target in right half
        else:  # Right half is sorted
            if nums[mid] < target <= nums[right]:
                left = mid + 1   # Target in right half
            else:
                right = mid - 1  # Target in left half
    
    return -1

def find_minimum_rotated(nums):
    """
    Find minimum element in rotated sorted array
    """
    left, right = 0, len(nums) - 1
    
    while left < right:
        mid = left + (right - left) // 2
        
        if nums[mid] > nums[right]:
            # Minimum is in right half
            left = mid + 1
        else:
            # Minimum is in left half (including mid)
            right = mid
    
    return nums[left]

def search_2d_matrix(matrix, target):
    """
    Search in 2D matrix (sorted row-wise and column-wise)
    """
    if not matrix or not matrix[0]:
        return False
    
    rows, cols = len(matrix), len(matrix[0])
    left, right = 0, rows * cols - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        mid_value = matrix[mid // cols][mid % cols]
        
        if mid_value == target:
            return True
        elif mid_value < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return False

def find_kth_smallest_matrix(matrix, k):
    """
    Find kth smallest element in sorted matrix
    """
    n = len(matrix)
    left, right = matrix[0][0], matrix[n-1][n-1]
    
    def count_less_equal(mid):
        count = 0
        j = n - 1  # Start from top-right corner
        
        for i in range(n):
            while j >= 0 and matrix[i][j] > mid:
                j -= 1
            count += j + 1
        
        return count
    
    while left < right:
        mid = left + (right - left) // 2
        
        if count_less_equal(mid) < k:
            left = mid + 1
        else:
            right = mid
    
    return left

def capacity_to_ship_packages(weights, days):
    """
    Find minimum capacity needed to ship packages within given days
    """
    def can_ship_in_days(capacity):
        days_needed = 1
        current_load = 0
        
        for weight in weights:
            if current_load + weight > capacity:
                days_needed += 1
                current_load = weight
            else:
                current_load += weight
        
        return days_needed <= days
    
    left = max(weights)  # Minimum possible capacity
    right = sum(weights)  # Maximum possible capacity
    
    while left < right:
        mid = left + (right - left) // 2
        
        if can_ship_in_days(mid):
            right = mid  # Try smaller capacity
        else:
            left = mid + 1  # Need larger capacity
    
    return left

def split_array_largest_sum(nums, m):
    """
    Split array into m subarrays to minimize largest sum
    """
    def can_split(max_sum):
        subarrays = 1
        current_sum = 0
        
        for num in nums:
            if current_sum + num > max_sum:
                subarrays += 1
                current_sum = num
            else:
                current_sum += num
        
        return subarrays <= m
    
    left = max(nums)    # Minimum possible largest sum
    right = sum(nums)   # Maximum possible largest sum
    
    while left < right:
        mid = left + (right - left) // 2
        
        if can_split(mid):
            right = mid
        else:
            left = mid + 1
    
    return left

def find_in_mountain_array(target, mountain_arr):
    """
    Search in mountain array (increases then decreases)
    """
    # First find the peak
    def find_peak():
        left, right = 0, mountain_arr.length() - 1
        
        while left < right:
            mid = left + (right - left) // 2
            
            if mountain_arr.get(mid) < mountain_arr.get(mid + 1):
                left = mid + 1
            else:
                right = mid
        
        return left
    
    # Binary search in increasing part
    def search_increasing(start, end):
        while start <= end:
            mid = start + (end - start) // 2
            val = mountain_arr.get(mid)
            
            if val == target:
                return mid
            elif val < target:
                start = mid + 1
            else:
                end = mid - 1
        
        return -1
    
    # Binary search in decreasing part
    def search_decreasing(start, end):
        while start <= end:
            mid = start + (end - start) // 2
            val = mountain_arr.get(mid)
            
            if val == target:
                return mid
            elif val > target:
                start = mid + 1
            else:
                end = mid - 1
        
        return -1
    
    peak = find_peak()
    
    # Search in increasing part first
    result = search_increasing(0, peak)
    if result != -1:
        return result
    
    # Search in decreasing part
    return search_decreasing(peak + 1, mountain_arr.length() - 1)
```

---

## 🔗 LeetCode Practice Problems (10)

- **Search in Rotated Sorted Array** – https://leetcode.com/problems/search-in-rotated-sorted-array/
- **Find Minimum in Rotated Sorted Array** – https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/
- **Search a 2D Matrix** – https://leetcode.com/problems/search-a-2d-matrix/
- **Kth Smallest Element in a Sorted Matrix** – https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/
- **Capacity To Ship Packages Within D Days** – https://leetcode.com/problems/capacity-to-ship-packages-within-d-days/
- **Split Array Largest Sum** – https://leetcode.com/problems/split-array-largest-sum/
- **Find in Mountain Array** – https://leetcode.com/problems/find-in-mountain-array/
- **Median of Two Sorted Arrays** – https://leetcode.com/problems/median-of-two-sorted-arrays/
- **Koko Eating Bananas** – https://leetcode.com/problems/koko-eating-bananas/
- **Minimum Number of Days to Make m Bouquets** – https://leetcode.com/problems/minimum-number-of-days-to-make-m-bouquets/

---

## 🧠 Flashcard (for spaced repetition)

What is the Modified Binary Search pattern? ::

• Adapts binary search for problems beyond simple sorted array searches (rotated arrays, 2D matrices, optimization)
• Maintains O(log n) by eliminating half the search space based on conditions
• Key is identifying the condition that allows eliminating half the possibilities

[[modified-binary-search]] 