# Dynamic Programming (1D)

## Explanation

1D Dynamic Programming solves problems by breaking them into smaller subproblems and storing results to avoid redundant calculations. It's used for optimization problems that exhibit optimal substructure and overlapping subproblems. The solution typically uses an array to store intermediate results and builds up to the final answer.

## Algorithm (Word-based)

- Define base cases
- Initialize DP array with base values
- Iterate through the problem space
- For each state, calculate value based on previously computed states
- Return the final result from the DP array

## Code Template

```python
def dp_1d_template(n):
    """
    Template for 1D dynamic programming
    """
    # Handle base cases
    if n <= base_case_threshold:
        return base_value
    
    # Initialize DP array
    dp = [0] * (n + 1)
    
    # Set base cases
    dp[0] = base_value_0  # or calculated base value
    dp[1] = base_value_1  # or calculated base value
    
    # Fill DP array iteratively
    for i in range(2, n + 1):
        # Calculate current state based on previous states
        dp[i] = calculate_from_previous(dp, i)
    
    return dp[n]

# Alternative: Space-optimized version
def dp_1d_optimized(n):
    """
    Space-optimized 1D DP when only previous values needed
    """
    if n <= 1:
        return n
    
    # Only keep track of necessary previous values
    prev2 = 0  # dp[i-2]
    prev1 = 1  # dp[i-1]
    
    for i in range(2, n + 1):
        current = calculate_from_previous(prev2, prev1, i)
        prev2 = prev1
        prev1 = current
    
    return prev1
```

## Practice Questions

1. [Climbing Stairs](https://leetcode.com/problems/climbing-stairs/) - Easy
2. [House Robber](https://leetcode.com/problems/house-robber/) - Medium
3. [Fibonacci Number](https://leetcode.com/problems/fibonacci-number/) - Easy
4. [Decode Ways](https://leetcode.com/problems/decode-ways/) - Medium
5. [Maximum Subarray](https://leetcode.com/problems/maximum-subarray/) - Easy
6. [House Robber II](https://leetcode.com/problems/house-robber-ii/) - Medium
7. [Decode Ways II](https://leetcode.com/problems/decode-ways-ii/) - Hard
8. [Paint House](https://leetcode.com/problems/paint-house/) - Medium
9. [Word Break](https://leetcode.com/problems/word-break/) - Medium
10. [Best Time to Buy and Sell Stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/) - Easy

## Notes

- Can often be optimized to O(1) space if only previous values needed
- Identify recurrence relation before implementation
- Look for overlapping subproblems and optimal substructure

---

**Tags:** #dsa #dp-1d #leetcode

**Last Reviewed:** 2025-11-05