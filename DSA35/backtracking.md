# Backtracking

## Explanation

Backtracking is a recursive algorithm that explores all possible solutions by building them incrementally and abandoning (backtracking) when a solution is no longer viable. It systematically searches the solution space by making choices, exploring consequences, and undoing choices to try alternatives.

## Algorithm (Word-based)

- Define base case to stop recursion
- Iterate through possible choices at current step
- Make a choice and add to current path
- Recursively explore with the choice
- Undo the choice (backtrack)
- Continue until all possibilities are exhausted

## Code Template

```python
def backtracking_template(problem_input):
    """
    General backtracking template
    """
    result = []
    path = []
    
    def backtrack(path, choices):
        # Base case - found a solution
        if is_solution(path):
            result.append(path[:])  # Add copy of current path
            return
        
        # Iterate through possible choices
        for choice in get_valid_choices(choices, path):
            # Make choice
            path.append(choice)
            
            # Recursively explore
            backtrack(path, get_remaining_choices(choices, choice))
            
            # Undo choice (backtrack)
            path.pop()
    
    backtrack(path, problem_input)
    return result

# Alternative template
def backtracking_solution(candidates):
    """
    Backtracking with specific constraints
    """
    def backtrack(start, current_path, remaining_target):
        # Check if current path is a solution
        if remaining_target == 0:
            result.append(current_path[:])
            return
        
        for i in range(start, len(candidates)):
            # Skip if candidate makes target negative
            if candidates[i] > remaining_target:
                continue
                
            # Make choice
            current_path.append(candidates[i])
            
            # Recurse with updated target
            backtrack(i, current_path, remaining_target - candidates[i])
            
            # Undo choice
            current_path.pop()
    
    result = []
    backtrack(0, [], target)
    return result
```

## Practice Questions

1. [Subsets](https://leetcode.com/problems/subsets/) - Medium
2. [Subsets II](https://leetcode.com/problems/subsets-ii/) - Medium
3. [Permutations](https://leetcode.com/problems/permutations/) - Medium
4. [Permutations II](https://leetcode.com/problems/permutations-ii/) - Medium
5. [Combinations](https://leetcode.com/problems/combinations/) - Medium
6. [Combination Sum](https://leetcode.com/problems/combination-sum/) - Medium
7. [Combination Sum II](https://leetcode.com/problems/combination-sum-ii/) - Medium
8. [Generate Parentheses](https://leetcode.com/problems/generate-parentheses/) - Medium
9. [Letter Combinations of a Phone Number](https://leetcode.com/problems/letter-combinations-of-a-phone-number/) - Medium
10. [N-Queens](https://leetcode.com/problems/n-queens/) - Hard

## Notes

- Essential for problems involving combinations, permutations, and possibilities
- Pruning invalid paths early improves efficiency
- Key concept is to explore and backtrack

---

**Tags:** #dsa #backtracking #leetcode

**Last Reviewed:** 2025-11-05