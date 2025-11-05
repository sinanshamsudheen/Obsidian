# Subsets

## Explanation

The subsets pattern generates all possible combinations of elements from a given set. It's based on the principle that for each element, we can either include it in the subset or exclude it. This creates a binary decision tree with 2^n possible subsets. It's essential for problems requiring exploration of all combinations.

## Algorithm (Word-based)

- Start with an empty subset in the result list
- For each element in the input
- Iterate through existing subsets in result
- Create a new subset by adding current element to each existing subset
- Add new subset to result
- Continue until all elements processed

## Code Template

```python
def generate_subsets(nums):
    """
    Generate all possible subsets of a set
    """
    result = [[]]  # Start with empty subset
    
    for num in nums:
        # For each existing subset, create a new one by adding current number
        new_subsets = []
        for subset in result:
            new_subset = subset + [num]
            new_subsets.append(new_subset)
        
        # Add all new subsets to result
        result.extend(new_subsets)
    
    return result

# Alternative: Backtracking approach
def generate_subsets_backtrack(nums):
    """
    Generate all subsets using backtracking
    """
    result = []
    
    def backtrack(start, current_subset):
        # Add current subset to result
        result.append(current_subset[:])
        
        # Try adding each remaining element
        for i in range(start, len(nums)):
            current_subset.append(nums[i])
            backtrack(i + 1, current_subset)
            current_subset.pop()  # Backtrack
    
    backtrack(0, [])
    return result

# Iterative approach
def generate_subsets_iterative(nums):
    """
    Generate subsets iteratively
    """
    subsets = [[]]
    
    for num in nums:
        # Extend by adding current number to each existing subset
        subsets += [subset + [num] for subset in subsets]
    
    return subsets
```

## Practice Questions

1. [Subsets](https://leetcode.com/problems/subsets/) - Medium
2. [Subsets II](https://leetcode.com/problems/subsets-ii/) - Medium
3. [Combination Sum](https://leetcode.com/problems/combination-sum/) - Medium
4. [Permutations](https://leetcode.com/problems/permutations/) - Medium
5. [Letter Case Permutation](https://leetcode.com/problems/letter-case-permutation/) - Medium
6. [Generate Parentheses](https://leetcode.com/problems/generate-parentheses/) - Medium
7. [Partition to K Equal Sum Subsets](https://leetcode.com/problems/partition-to-k-equal-sum-subsets/) - Medium
8. [Target Sum](https://leetcode.com/problems/target-sum/) - Medium
9. [Partition Equal Subset Sum](https://leetcode.com/problems/partition-equal-subset-sum/) - Medium
10. [Binary Tree Paths](https://leetcode.com/problems/binary-tree-paths/) - Easy

## Notes

- Time complexity: O(2^n) since there are 2^n possible subsets
- Can handle duplicates with careful sorting and skipping
- Useful as a base for many combination problems

---

**Tags:** #dsa #subsets #leetcode

**Last Reviewed:** 2025-11-05