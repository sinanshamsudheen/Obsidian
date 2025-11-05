# Permutations

## Explanation

The permutations pattern generates all possible arrangements of elements in a given set. Unlike subsets, permutations consider the order of elements. For n elements, there are n! possible permutations. This pattern is used when the order matters, such as arranging items or finding all possible sequences.

## Algorithm (Word-based)

- For each position in the result
- Try placing each available element
- Mark element as used
- Recursively generate permutations for remaining positions
- Mark element as unused (backtrack)
- Continue until all positions filled

## Code Template

```python
def generate_permutations(nums):
    """
    Generate all possible permutations of a list
    """
    result = []
    
    def backtrack(current_permutation):
        # Base case: if permutation is complete, add to result
        if len(current_permutation) == len(nums):
            result.append(current_permutation[:])
            return
        
        # Try each unused number
        for num in nums:
            if num not in current_permutation:
                current_permutation.append(num)
                backtrack(current_permutation)
                current_permutation.pop()  # Backtrack
    
    backtrack([])
    return result

# Alternative with used array
def generate_permutations_optimized(nums):
    """
    Generate permutations using a boolean array to track used elements
    """
    result = []
    used = [False] * len(nums)
    
    def backtrack(current_permutation):
        if len(current_permutation) == len(nums):
            result.append(current_permutation[:])
            return
        
        for i in range(len(nums)):
            if not used[i]:
                used[i] = True
                current_permutation.append(nums[i])
                backtrack(current_permutation)
                current_permutation.pop()
                used[i] = False  # Backtrack
    
    backtrack([])
    return result

# Handle duplicates
def generate_permutations_unique(nums):
    """
    Generate unique permutations (handles duplicates)
    """
    result = []
    used = [False] * len(nums)
    nums.sort()  # Sort to handle duplicates
    
    def backtrack(current_permutation):
        if len(current_permutation) == len(nums):
            result.append(current_permutation[:])
            return
        
        for i in range(len(nums)):
            if used[i]:
                continue
            
            # Skip duplicates: if current equals previous and previous is not used
            if i > 0 and nums[i] == nums[i-1] and not used[i-1]:
                continue
            
            used[i] = True
            current_permutation.append(nums[i])
            backtrack(current_permutation)
            current_permutation.pop()
            used[i] = False
    
    backtrack([])
    return result
```

## Practice Questions

1. [Permutations](https://leetcode.com/problems/permutations/) - Medium
2. [Permutations II](https://leetcode.com/problems/permutations-ii/) - Medium
3. [Next Permutation](https://leetcode.com/problems/next-permutation/) - Medium
4. [Permutation Sequence](https://leetcode.com/problems/permutation-sequence/) - Hard
5. [Letter Tile Possibilities](https://leetcode.com/problems/letter-tile-possibilities/) - Medium
6. [Palindrome Permutation II](https://leetcode.com/problems/palindrome-permutation-ii/) - Medium
7. [Beautiful Arrangement](https://leetcode.com/problems/beautiful-arrangement/) - Medium
8. [Letter Combinations of a Phone Number](https://leetcode.com/problems/letter-combinations-of-a-phone-number/) - Medium
9. [Brick Wall](https://leetcode.com/problems/brick-wall/) - Medium
10. [N-Queens](https://leetcode.com/problems/n-queens/) - Hard

## Notes

- Time complexity: O(n! * n) for generating all permutations
- Can handle duplicates with careful checking
- Order matters in permutations, unlike subsets

---

**Tags:** #dsa #permutations #leetcode

**Last Reviewed:** 2025-11-05