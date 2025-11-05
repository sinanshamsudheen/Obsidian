# Monotonic Stack

## Explanation

The Monotonic Stack pattern uses a stack that maintains elements in either increasing or decreasing order. It's particularly effective for problems where we need to find the next greater or smaller element for each item in an array. The stack helps track elements that haven't found their "next" element yet.

## Algorithm (Word-based)

- Initialize empty stack
- Iterate through array elements
- While current element breaks monotonic property
- Process elements being popped from stack
- Push current element to maintain monotonic property
- Return results based on problem requirements

## Code Template

```python
def monotonic_stack_increasing(arr):
    """
    Find next greater element for each element in array
    """
    result = [-1] * len(arr)
    stack = []  # Monotonic decreasing stack (holds indices)
    
    for i in range(len(arr)):
        # While stack is not empty and current element is greater than top
        while stack and arr[stack[-1]] < arr[i]:
            index = stack.pop()
            result[index] = arr[i]  # Current element is next greater
        
        stack.append(i)
    
    return result

def monotonic_stack_decreasing(arr):
    """
    Find next smaller element for each element in array
    """
    result = [-1] * len(arr)
    stack = []  # Monotonic increasing stack (holds indices)
    
    for i in range(len(arr)):
        # While stack is not empty and current element is smaller than top
        while stack and arr[stack[-1]] > arr[i]:
            index = stack.pop()
            result[index] = arr[i]  # Current element is next smaller
        
        stack.append(i)
    
    return result

def largest_rectangle_histogram(heights):
    """
    Monotonic stack application: largest rectangle in histogram
    """
    stack = []  # Store indices in increasing order of heights
    max_area = 0
    
    for i in range(len(heights) + 1):
        current_height = 0 if i == len(heights) else heights[i]
        
        while stack and heights[stack[-1]] > current_height:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, height * width)
        
        stack.append(i)
    
    return max_area
```

## Practice Questions

1. [Next Greater Element I](https://leetcode.com/problems/next-greater-element-i/) - Easy
2. [Next Greater Element II](https://leetcode.com/problems/next-greater-element-ii/) - Medium
3. [Daily Temperatures](https://leetcode.com/problems/daily-temperatures/) - Medium
4. [Largest Rectangle in Histogram](https://leetcode.com/problems/largest-rectangle-in-histogram/) - Hard
5. [Trapping Rain Water](https://leetcode.com/problems/trapping-rain-water/) - Hard
6. [Remove K Digits](https://leetcode.com/problems/remove-k-digits/) - Medium
7. [Create Maximum Number](https://leetcode.com/problems/create-maximum-number/) - Hard
8. [Maximal Rectangle](https://leetcode.com/problems/maximal-rectangle/) - Hard
9. [Remove Duplicate Letters](https://leetcode.com/problems/remove-duplicate-letters/) - Medium
10. [Sum of Subarray Minimums](https://leetcode.com/problems/sum-of-subarray-minimums/) - Medium

## Notes

- Use decreasing stack for next greater elements
- Use increasing stack for next smaller elements
- Stack typically stores indices rather than values
- Useful for range-based problems

---

**Tags:** #dsa #monotonic-stack #stack #leetcode

**Last Reviewed:** 2025-11-05