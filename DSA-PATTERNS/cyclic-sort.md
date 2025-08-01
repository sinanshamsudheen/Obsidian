## 🧠 Cyclic Sort

The **Cyclic Sort** pattern is used when dealing with arrays containing numbers in a given range, typically from 1 to n.

• Places each number at its correct index (number 1 goes to index 0, number 2 to index 1, etc.)
• Achieves O(n) time complexity and O(1) space complexity
• Useful for finding missing numbers, duplicates, or first missing positive
• Works by swapping elements until each number is in its correct position

**Related:** [[Array]] [[Sorting]]

---

## 💡 Python Code Snippet

```python
def cyclic_sort(nums):
    """
    Sort array containing numbers from 1 to n using cyclic sort
    Each number should be at index (number - 1)
    """
    i = 0
    while i < len(nums):
        # Calculate correct index for current number
        correct_index = nums[i] - 1
        
        # If number is not at its correct position, swap it
        if nums[i] != nums[correct_index]:
            nums[i], nums[correct_index] = nums[correct_index], nums[i]
        else:
            i += 1
    
    return nums

def find_missing_number(nums):
    """
    Find missing number in array containing numbers from 0 to n
    """
    n = len(nums)
    i = 0
    
    # Place each number at its correct index
    while i < n:
        if nums[i] < n and nums[i] != nums[nums[i]]:
            nums[nums[i]], nums[i] = nums[i], nums[nums[i]]
        else:
            i += 1
    
    # Find the first number not at its correct position
    for i in range(n):
        if nums[i] != i:
            return i
    
    return n  # If all numbers 0 to n-1 are present

def find_all_missing_numbers(nums):
    """
    Find all missing numbers in array containing numbers from 1 to n
    """
    i = 0
    while i < len(nums):
        correct_index = nums[i] - 1
        if nums[i] != nums[correct_index]:
            nums[i], nums[correct_index] = nums[correct_index], nums[i]
        else:
            i += 1
    
    missing_numbers = []
    for i in range(len(nums)):
        if nums[i] != i + 1:
            missing_numbers.append(i + 1)
    
    return missing_numbers

def find_duplicate(nums):
    """
    Find the duplicate number in array containing numbers from 1 to n
    """
    i = 0
    while i < len(nums):
        if nums[i] != i + 1:
            correct_index = nums[i] - 1
            if nums[i] != nums[correct_index]:
                nums[i], nums[correct_index] = nums[correct_index], nums[i]
            else:
                return nums[i]  # Found duplicate
        else:
            i += 1
    
    return -1
```

---

## 🔗 LeetCode Practice Problems (10)

- **Missing Number** – https://leetcode.com/problems/missing-number/
- **Find All Numbers Disappeared in an Array** – https://leetcode.com/problems/find-all-numbers-disappeared-in-an-array/
- **Find the Duplicate Number** – https://leetcode.com/problems/find-the-duplicate-number/
- **First Missing Positive** – https://leetcode.com/problems/first-missing-positive/
- **Find All Duplicates in an Array** – https://leetcode.com/problems/find-all-duplicates-in-an-array/
- **Set Mismatch** – https://leetcode.com/problems/set-mismatch/
- **Find the Smallest Missing Positive Number** – https://leetcode.com/problems/first-missing-positive/
- **Couples Holding Hands** – https://leetcode.com/problems/couples-holding-hands/
- **Array Nesting** – https://leetcode.com/problems/array-nesting/
- **Missing Element in Sorted Array** – https://leetcode.com/problems/missing-element-in-sorted-array/

---

## 🧠 Flashcard (for spaced repetition)

What is the Cyclic Sort pattern? ::

• Places each number at its correct index in array containing numbers from 1 to n
• Achieves O(n) time and O(1) space by swapping elements to correct positions  
• Used for finding missing numbers, duplicates, or first missing positive in arrays

[[cyclic-sort]] 