# Stack Problems

## Explanation

Stack problems leverage the Last-In-First-Out (LIFO) property of stacks for various algorithms. Stacks are ideal for problems involving nested structures like parentheses, function calls, or when you need to access the most recent element. This pattern appears in expression evaluation, backtracking, and various algorithmic problems.

## Algorithm (Word-based)

- Identify if problem involves nested or hierarchical structures
- Use stack to keep track of elements or states
- Push elements when certain conditions are met
- Pop elements when they can be processed
- Maintain order and hierarchy using stack properties

## Code Template

```python
def is_valid_parentheses(s):
    """
    Check if parentheses string is valid using stack
    """
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    
    for char in s:
        if char in mapping.values():
            # Opening bracket, push to stack
            stack.append(char)
        elif char in mapping.keys():
            # Closing bracket
            if not stack or stack.pop() != mapping[char]:
                return False
        # Ignore other characters if any
    
    return not stack  # Stack should be empty

def evaluate_expression(s):
    """
    Evaluate mathematical expression with +, -, (, )
    """
    stack = []
    num = 0
    result = 0
    sign = 1  # 1 for positive, -1 for negative
    
    for char in s:
        if char.isdigit():
            num = num * 10 + int(char)
        elif char == '+':
            result += sign * num
            num = 0
            sign = 1
        elif char == '-':
            result += sign * num
            num = 0
            sign = -1
        elif char == '(':
            # Push current result and sign to stack
            stack.append(result)
            stack.append(sign)
            # Reset for sub-expression
            result = 0
            sign = 1
        elif char == ')':
            result += sign * num
            num = 0
            # Retrieve sign and previous result
            result *= stack.pop()  # Apply sign before parenthesis
            result += stack.pop()  # Add previous result
    
    result += sign * num
    return result

def next_greater_element(nums):
    """
    Find next greater element for each element using stack
    """
    stack = []
    result = [-1] * len(nums)
    
    for i in range(len(nums) - 1, -1, -1):  # Process from right to left
        # Pop elements smaller than current
        while stack and stack[-1] <= nums[i]:
            stack.pop()
        
        if stack:
            result[i] = stack[-1]
        
        stack.append(nums[i])
    
    return result

def simplify_path(path):
    """
    Simplify Unix-style absolute path
    """
    stack = []
    parts = path.split('/')
    
    for part in parts:
        if part == '..':
            # Go up one directory level
            if stack:
                stack.pop()
        elif part != '' and part != '.':
            # Add valid directory names
            stack.append(part)
    
    return '/' + '/'.join(stack)
```

## Practice Questions

1. [Valid Parentheses](https://leetcode.com/problems/valid-parentheses/) - Easy
2. [Min Stack](https://leetcode.com/problems/min-stack/) - Medium
3. [Evaluate Reverse Polish Notation](https://leetcode.com/problems/evaluate-reverse-polish-notation/) - Medium
4. [Generate Parentheses](https://leetcode.com/problems/generate-parentheses/) - Medium
5. [Daily Temperatures](https://leetcode.com/problems/daily-temperatures/) - Medium
6. [Next Greater Element I](https://leetcode.com/problems/next-greater-element-i/) - Easy
7. [Decode String](https://leetcode.com/problems/decode-string/) - Medium
8. [Basic Calculator](https://leetcode.com/problems/basic-calculator/) - Hard
9. [Remove K Digits](https://leetcode.com/problems/remove-k-digits/) - Medium
10. [Flatten Nested List Iterator](https://leetcode.com/problems/flatten-nested-list-iterator/) - Medium

## Notes

- Stacks are perfect for processing nested structures
- Useful for implementing iterators for nested data
- Can help maintain state in recursive algorithms

---

**Tags:** #dsa #stack #lifo #parentheses #leetcode

**Last Reviewed:** 2025-11-05