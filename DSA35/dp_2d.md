# Dynamic Programming (2D)

## Explanation

2D Dynamic Programming solves problems using a 2D array where each cell represents a subproblem based on two variables. It's used for problems involving sequences, matrices, or when state depends on two parameters. Common applications include sequence alignment, game theory, and optimization with multiple constraints.

## Algorithm (Word-based)

- Define base cases for the 2D state
- Initialize DP table with base values
- Iterate through both dimensions
- For each cell, calculate value based on neighboring cells
- Return the final result from the appropriate cell

## Code Template

```python
def dp_2d_template(m, n):
    """
    Template for 2D dynamic programming
    """
    # Create 2D DP table
    dp = [[0] * n for _ in range(m)]
    
    # Initialize base cases
    # First row
    for j in range(n):
        dp[0][j] = base_case_first_row(j)
    
    # First column
    for i in range(m):
        dp[i][0] = base_case_first_col(i)
    
    # Fill the DP table
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = calculate_state(dp, i, j)
    
    return dp[m-1][n-1]

# Alternative: Bottom-up approach
def dp_2d_bottom_up(grid):
    """
    2D DP with grid-based problem
    """
    rows, cols = len(grid), len(grid[0])
    
    # Initialize DP table
    dp = [[0] * cols for _ in range(rows)]
    
    # Set base case
    dp[0][0] = grid[0][0]
    
    # Fill first row
    for j in range(1, cols):
        dp[0][j] = dp[0][j-1] + grid[0][j]
    
    # Fill first column
    for i in range(1, rows):
        dp[i][0] = dp[i-1][0] + grid[i][0]
    
    # Fill remaining cells
    for i in range(1, rows):
        for j in range(1, cols):
            dp[i][j] = calculate_from_neighbors(dp, grid, i, j)
    
    return dp[rows-1][cols-1]
```

## Practice Questions

1. [Unique Paths](https://leetcode.com/problems/unique-paths/) - Medium
2. [Unique Paths II](https://leetcode.com/problems/unique-paths-ii/) - Medium
3. [Minimum Path Sum](https://leetcode.com/problems/minimum-path-sum/) - Medium
4. [Longest Common Subsequence](https://leetcode.com/problems/longest-common-subsequence/) - Medium
5. [Edit Distance](https://leetcode.com/problems/edit-distance/) - Medium
6. [Best Time to Buy and Sell Stock III](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/) - Hard
7. [Distinct Subsequences](https://leetcode.com/problems/distinct-subsequences/) - Hard
8. [Interleaving String](https://leetcode.com/problems/interleaving-string/) - Medium
9. [Regular Expression Matching](https://leetcode.com/problems/regular-expression-matching/) - Hard
10. [Wildcard Matching](https://leetcode.com/problems/wildcard-matching/) - Hard

## Notes

- Can sometimes be optimized to 1D if only previous row/column needed
- Key is identifying the recurrence relation between states
- Often requires careful initialization of base cases

---

**Tags:** #dsa #dp-2d #leetcode

**Last Reviewed:** 2025-11-05