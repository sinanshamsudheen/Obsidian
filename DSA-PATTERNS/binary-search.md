## 🧠 Binary Search

**Binary Search** efficiently finds a target value in a sorted array by repeatedly dividing the search space in half.

• Works only on sorted data structures
• Eliminates half of remaining elements in each iteration
• Time complexity: O(log n), Space complexity: O(1) for iterative approach
• Key insight: compare middle element with target to decide which half to search

**Related:** [[Array]] [[Sorting]] [[Divide and Conquer]]

---

## 💡 Python Code Snippet

```python
def binary_search_iterative(arr, target):
    """
    Standard binary search - find index of target in sorted array
    """
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = left + (right - left) // 2  # Avoid overflow
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1  # Search right half
        else:
            right = mid - 1  # Search left half
    
    return -1  # Target not found

def binary_search_recursive(arr, target, left=0, right=None):
    """
    Recursive binary search
    """
    if right is None:
        right = len(arr) - 1
    
    if left > right:
        return -1
    
    mid = left + (right - left) // 2
    
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)

def search_insert_position(arr, target):
    """
    Find position where target should be inserted to maintain sorted order
    """
    left, right = 0, len(arr)
    
    while left < right:
        mid = left + (right - left) // 2
        
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid
    
    return left

def find_first_occurrence(arr, target):
    """
    Find first occurrence of target in sorted array with duplicates
    """
    left, right = 0, len(arr) - 1
    result = -1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            result = mid
            right = mid - 1  # Continue searching left for first occurrence
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return result

def find_last_occurrence(arr, target):
    """
    Find last occurrence of target in sorted array with duplicates
    """
    left, right = 0, len(arr) - 1
    result = -1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            result = mid
            left = mid + 1  # Continue searching right for last occurrence
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return result

def search_range(arr, target):
    """
    Find first and last position of target in sorted array
    """
    first = find_first_occurrence(arr, target)
    if first == -1:
        return [-1, -1]
    
    last = find_last_occurrence(arr, target)
    return [first, last]

def find_peak_element(arr):
    """
    Find peak element (greater than neighbors) in array
    """
    left, right = 0, len(arr) - 1
    
    while left < right:
        mid = left + (right - left) // 2
        
        if arr[mid] > arr[mid + 1]:
            # Peak is in left half (including mid)
            right = mid
        else:
            # Peak is in right half
            left = mid + 1
    
    return left

def sqrt_binary_search(x):
    """
    Find square root using binary search (integer part)
    """
    if x < 2:
        return x
    
    left, right = 1, x // 2
    
    while left <= right:
        mid = left + (right - left) // 2
        square = mid * mid
        
        if square == x:
            return mid
        elif square < x:
            left = mid + 1
        else:
            right = mid - 1
    
    return right  # Largest integer whose square <= x
```

---

## 🔗 LeetCode Practice Problems (10)

- **Binary Search** – https://leetcode.com/problems/binary-search/
- **Search Insert Position** – https://leetcode.com/problems/search-insert-position/
- **Find First and Last Position of Element in Sorted Array** – https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/
- **Search in Rotated Sorted Array** – https://leetcode.com/problems/search-in-rotated-sorted-array/
- **Find Peak Element** – https://leetcode.com/problems/find-peak-element/
- **Sqrt(x)** – https://leetcode.com/problems/sqrtx/
- **Search a 2D Matrix** – https://leetcode.com/problems/search-a-2d-matrix/
- **Find Minimum in Rotated Sorted Array** – https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/
- **Single Element in a Sorted Array** – https://leetcode.com/problems/single-element-in-a-sorted-array/
- **Kth Smallest Element in a Sorted Matrix** – https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/

---

## 🧠 Flashcard (for spaced repetition)

What is the Binary Search pattern? :: • Efficiently finds target in sorted array by repeatedly dividing search space in half • O(log n) time complexity by eliminating half of elements each iteration • Compares middle element with target to decide which half to search next [[binary-search]] 