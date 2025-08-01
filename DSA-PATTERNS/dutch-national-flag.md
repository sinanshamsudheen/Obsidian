## 🧠 Dutch National Flag

**Dutch National Flag** efficiently partitions an array into three sections based on a pivot value, using three pointers to achieve O(n) time and O(1) space.

• Partitions array into three parts: elements less than pivot, equal to pivot, and greater than pivot
• Uses three pointers: low (boundary of elements < pivot), mid (current element), high (boundary of elements > pivot)
• Single pass algorithm with constant extra space
• Named after Dutch flag colors: red, white, blue (or 0, 1, 2)

**Related:** [[Array]] [[Sorting]] [[Two Pointers]] [[Partitioning]]

---

## 💡 Python Code Snippet

```python
def sort_colors(nums):
    """
    Classic Dutch National Flag: sort array of 0s, 1s, and 2s
    """
    low = mid = 0
    high = len(nums) - 1
    
    while mid <= high:
        if nums[mid] == 0:
            # Swap with low boundary and advance both pointers
            nums[low], nums[mid] = nums[mid], nums[low]
            low += 1
            mid += 1
        elif nums[mid] == 1:
            # Correct position, just advance mid
            mid += 1
        else:  # nums[mid] == 2
            # Swap with high boundary, advance high, don't advance mid
            # (need to check the swapped element)
            nums[mid], nums[high] = nums[high], nums[mid]
            high -= 1
    
    return nums

def partition_array_three_way(nums, pivot):
    """
    Partition array into three parts based on pivot value
    """
    low = mid = 0
    high = len(nums) - 1
    
    while mid <= high:
        if nums[mid] < pivot:
            nums[low], nums[mid] = nums[mid], nums[low]
            low += 1
            mid += 1
        elif nums[mid] == pivot:
            mid += 1
        else:  # nums[mid] > pivot
            nums[mid], nums[high] = nums[high], nums[mid]
            high -= 1
    
    return nums

def sort_array_by_parity(nums):
    """
    Sort array so that even numbers come before odd numbers
    """
    left = 0
    right = len(nums) - 1
    
    while left < right:
        # Find odd number from left
        while left < right and nums[left] % 2 == 0:
            left += 1
        
        # Find even number from right
        while left < right and nums[right] % 2 == 1:
            right -= 1
        
        # Swap if both found
        if left < right:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
            right -= 1
    
    return nums

def sort_array_by_parity_ii(nums):
    """
    Sort array so that even-indexed positions have even numbers
    and odd-indexed positions have odd numbers
    """
    even_idx = 0  # Points to next even index
    odd_idx = 1   # Points to next odd index
    n = len(nums)
    
    while even_idx < n and odd_idx < n:
        # If even index has correct parity, advance
        if nums[even_idx] % 2 == 0:
            even_idx += 2
        # If odd index has correct parity, advance
        elif nums[odd_idx] % 2 == 1:
            odd_idx += 2
        # Both have wrong parity, swap them
        else:
            nums[even_idx], nums[odd_idx] = nums[odd_idx], nums[even_idx]
            even_idx += 2
            odd_idx += 2
    
    return nums

def move_zeros_to_end(nums):
    """
    Move all zeros to end while maintaining relative order of non-zeros
    """
    write_idx = 0
    
    # Move all non-zero elements to the front
    for read_idx in range(len(nums)):
        if nums[read_idx] != 0:
            nums[write_idx] = nums[read_idx]
            write_idx += 1
    
    # Fill remaining positions with zeros
    while write_idx < len(nums):
        nums[write_idx] = 0
        write_idx += 1
    
    return nums

def segregate_positive_negative(nums):
    """
    Segregate positive and negative numbers (order may change)
    """
    left = 0
    right = len(nums) - 1
    
    while left < right:
        # Find negative number from left
        while left < right and nums[left] >= 0:
            left += 1
        
        # Find positive number from right
        while left < right and nums[right] < 0:
            right -= 1
        
        # Swap if both found
        if left < right:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
            right -= 1
    
    return nums

def three_way_quicksort(nums, low, high):
    """
    Three-way quicksort using Dutch National Flag partitioning
    """
    if low >= high:
        return
    
    # Choose pivot (middle element for simplicity)
    pivot = nums[(low + high) // 2]
    
    # Dutch National Flag partitioning
    i = low  # Current element
    j = low  # Boundary of elements < pivot
    k = high # Boundary of elements > pivot
    
    while i <= k:
        if nums[i] < pivot:
            nums[i], nums[j] = nums[j], nums[i]
            i += 1
            j += 1
        elif nums[i] > pivot:
            nums[i], nums[k] = nums[k], nums[i]
            k -= 1
            # Don't increment i (need to check swapped element)
        else:  # nums[i] == pivot
            i += 1
    
    # Recursively sort the two partitions
    three_way_quicksort(nums, low, j - 1)
    three_way_quicksort(nums, k + 1, high)

def partition_around_pivot(nums, pivot_idx):
    """
    Partition array around given pivot index
    """
    pivot_val = nums[pivot_idx]
    
    # Move pivot to end
    nums[pivot_idx], nums[-1] = nums[-1], nums[pivot_idx]
    
    # Partition
    smaller = 0
    for i in range(len(nums) - 1):
        if nums[i] < pivot_val:
            nums[i], nums[smaller] = nums[smaller], nums[i]
            smaller += 1
    
    # Place pivot in correct position
    nums[smaller], nums[-1] = nums[-1], nums[smaller]
    
    return smaller  # Return final position of pivot

def sort_012_without_sorting_algorithm(nums):
    """
    Alternative implementation of Dutch National Flag
    """
    count = [0, 0, 0]  # Count of 0s, 1s, 2s
    
    # Count occurrences
    for num in nums:
        count[num] += 1
    
    # Overwrite array
    idx = 0
    for value in range(3):
        for _ in range(count[value]):
            nums[idx] = value
            idx += 1
    
    return nums

def wiggle_sort(nums):
    """
    Rearrange array so that nums[0] < nums[1] > nums[2] < nums[3]...
    """
    for i in range(1, len(nums)):
        if (i % 2 == 1 and nums[i] < nums[i-1]) or \
           (i % 2 == 0 and nums[i] > nums[i-1]):
            nums[i], nums[i-1] = nums[i-1], nums[i]
    
    return nums

def separate_even_odd_preserve_order(nums):
    """
    Separate even and odd numbers while preserving relative order
    """
    even_nums = []
    odd_nums = []
    
    # Separate into two lists
    for num in nums:
        if num % 2 == 0:
            even_nums.append(num)
        else:
            odd_nums.append(num)
    
    # Combine: even numbers first, then odd numbers
    return even_nums + odd_nums

def dutch_flag_generic(nums, compare_func):
    """
    Generic Dutch National Flag with custom comparison function
    compare_func returns -1, 0, or 1 for less than, equal to, or greater than pivot
    """
    low = mid = 0
    high = len(nums) - 1
    
    while mid <= high:
        comparison = compare_func(nums[mid])
        
        if comparison < 0:  # Less than pivot
            nums[low], nums[mid] = nums[mid], nums[low]
            low += 1
            mid += 1
        elif comparison == 0:  # Equal to pivot
            mid += 1
        else:  # Greater than pivot
            nums[mid], nums[high] = nums[high], nums[mid]
            high -= 1
    
    return nums

def kth_largest_using_partition(nums, k):
    """
    Find kth largest element using Dutch National Flag partitioning
    """
    def partition(left, right, pivot_idx):
        pivot_val = nums[pivot_idx]
        
        # Move pivot to end
        nums[pivot_idx], nums[right] = nums[right], nums[pivot_idx]
        
        store_idx = left
        for i in range(left, right):
            if nums[i] > pivot_val:  # For kth largest, we want descending order
                nums[store_idx], nums[i] = nums[i], nums[store_idx]
                store_idx += 1
        
        # Place pivot in final position
        nums[right], nums[store_idx] = nums[store_idx], nums[right]
        return store_idx
    
    def quickselect(left, right, k_smallest):
        if left == right:
            return nums[left]
        
        # Choose random pivot
        pivot_idx = left
        pivot_idx = partition(left, right, pivot_idx)
        
        if k_smallest == pivot_idx:
            return nums[k_smallest]
        elif k_smallest < pivot_idx:
            return quickselect(left, pivot_idx - 1, k_smallest)
        else:
            return quickselect(pivot_idx + 1, right, k_smallest)
    
    return quickselect(0, len(nums) - 1, k - 1)
```

---

## 🔗 LeetCode Practice Problems (10)

- **Sort Colors** – https://leetcode.com/problems/sort-colors/
- **Sort Array By Parity** – https://leetcode.com/problems/sort-array-by-parity/
- **Sort Array By Parity II** – https://leetcode.com/problems/sort-array-by-parity-ii/
- **Move Zeroes** – https://leetcode.com/problems/move-zeroes/
- **Wiggle Sort** – https://leetcode.com/problems/wiggle-sort/
- **Kth Largest Element in an Array** – https://leetcode.com/problems/kth-largest-element-in-an-array/
- **Partition Array into Three Parts with Equal Sum** – https://leetcode.com/problems/partition-array-into-three-parts-with-equal-sum/
- **Squares of a Sorted Array** – https://leetcode.com/problems/squares-of-a-sorted-array/
- **Partition Labels** – https://leetcode.com/problems/partition-labels/
- **3Sum** – https://leetcode.com/problems/3sum/

---

## 🧠 Flashcard (for spaced repetition)

What is the Dutch National Flag pattern? ::

• Efficiently partitions array into three sections using three pointers in O(n) time and O(1) space
• Uses low (< pivot boundary), mid (current), high (> pivot boundary) pointers
• Applied to sorting colors, partitioning arrays, and three-way quicksort problems

[[dutch-national-flag]] 