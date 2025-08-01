## 🧠 0/1 Knapsack (DP)

**0/1 Knapsack** is a classic dynamic programming problem where each item can be either included (1) or excluded (0) from the knapsack.

• Given items with weights and values, maximize value within weight capacity
• Each item can be taken at most once (0 or 1 times)
• Two approaches: 2D DP table or space-optimized 1D array
• Time complexity: O(n × W), Space: O(W) with optimization

**Related:** [[Dynamic Programming]] [[Array]] [[Optimization]]

---

## 💡 Python Code Snippet

```python
def knapsack_01_2d(weights, values, capacity):
    """
    0/1 Knapsack using 2D DP table
    dp[i][w] = maximum value using first i items with weight limit w
    """
    n = len(weights)
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            # Don't include current item
            dp[i][w] = dp[i-1][w]
            
            # Include current item if it fits
            if weights[i-1] <= w:
                include_value = values[i-1] + dp[i-1][w - weights[i-1]]
                dp[i][w] = max(dp[i][w], include_value)
    
    return dp[n][capacity]

def knapsack_01_1d(weights, values, capacity):
    """
    0/1 Knapsack using space-optimized 1D DP
    """
    dp = [0] * (capacity + 1)
    
    for i in range(len(weights)):
        # Traverse backwards to avoid using updated values
        for w in range(capacity, weights[i] - 1, -1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    
    return dp[capacity]

def knapsack_with_items(weights, values, capacity):
    """
    0/1 Knapsack that also returns selected items
    """
    n = len(weights)
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
    
    # Fill DP table
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            dp[i][w] = dp[i-1][w]
            
            if weights[i-1] <= w:
                include_value = values[i-1] + dp[i-1][w - weights[i-1]]
                dp[i][w] = max(dp[i][w], include_value)
    
    # Backtrack to find selected items
    w = capacity
    selected_items = []
    
    for i in range(n, 0, -1):
        # If value came from including current item
        if dp[i][w] != dp[i-1][w]:
            selected_items.append(i-1)  # Add item index
            w -= weights[i-1]
    
    return dp[n][capacity], selected_items[::-1]

def subset_sum(nums, target):
    """
    Check if subset exists with given sum (0/1 knapsack variant)
    """
    dp = [False] * (target + 1)
    dp[0] = True  # Empty subset has sum 0
    
    for num in nums:
        # Traverse backwards
        for j in range(target, num - 1, -1):
            dp[j] = dp[j] or dp[j - num]
    
    return dp[target]

def can_partition_equal_sum(nums):
    """
    Check if array can be partitioned into two equal sum subsets
    """
    total_sum = sum(nums)
    
    # If total sum is odd, can't partition equally
    if total_sum % 2 != 0:
        return False
    
    target = total_sum // 2
    return subset_sum(nums, target)

def count_subset_sum(nums, target):
    """
    Count number of subsets with given sum
    """
    dp = [0] * (target + 1)
    dp[0] = 1  # One way to make sum 0 (empty subset)
    
    for num in nums:
        for j in range(target, num - 1, -1):
            dp[j] += dp[j - num]
    
    return dp[target]

def minimum_subset_sum_difference(nums):
    """
    Find minimum difference between two subset sums
    """
    total_sum = sum(nums)
    
    # Find all possible subset sums up to total_sum // 2
    dp = [False] * (total_sum // 2 + 1)
    dp[0] = True
    
    for num in nums:
        for j in range(total_sum // 2, num - 1, -1):
            dp[j] = dp[j] or dp[j - num]
    
    # Find largest sum <= total_sum // 2
    sum1 = 0
    for j in range(total_sum // 2, -1, -1):
        if dp[j]:
            sum1 = j
            break
    
    sum2 = total_sum - sum1
    return abs(sum2 - sum1)

def target_sum_ways(nums, target):
    """
    Count ways to assign + or - to each number to reach target
    Transform to subset sum: find subset with sum = (target + total_sum) / 2
    """
    total_sum = sum(nums)
    
    # Check if transformation is valid
    if target > total_sum or target < -total_sum or (target + total_sum) % 2 != 0:
        return 0
    
    subset_sum_target = (target + total_sum) // 2
    return count_subset_sum(nums, subset_sum_target)

def ones_and_zeros(strs, m, n):
    """
    Maximum number of strings in subset with at most m zeros and n ones
    3D knapsack: items=strings, constraints=zeros and ones
    """
    # dp[i][j] = max strings with at most i zeros and j ones
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
    
    for s in strs:
        zeros = s.count('0')
        ones = s.count('1')
        
        # Traverse backwards to avoid using updated values
        for i in range(m, zeros - 1, -1):
            for j in range(n, ones - 1, -1):
                dp[i][j] = max(dp[i][j], dp[i - zeros][j - ones] + 1)
    
    return dp[m][n]

def coin_change_ways(coins, amount):
    """
    Number of ways to make amount using coins (0/1 knapsack - each coin used once)
    """
    dp = [0] * (amount + 1)
    dp[0] = 1
    
    for coin in coins:
        for j in range(coin, amount + 1):
            dp[j] += dp[j - coin]
    
    return dp[amount]
```

---

## 🔗 LeetCode Practice Problems (10)

- **Partition Equal Subset Sum** – https://leetcode.com/problems/partition-equal-subset-sum/
- **Target Sum** – https://leetcode.com/problems/target-sum/
- **Subset Sum** – https://leetcode.com/problems/partition-equal-subset-sum/
- **Ones and Zeroes** – https://leetcode.com/problems/ones-and-zeroes/
- **Last Stone Weight II** – https://leetcode.com/problems/last-stone-weight-ii/
- **Combination Sum IV** – https://leetcode.com/problems/combination-sum-iv/
- **Perfect Squares** – https://leetcode.com/problems/perfect-squares/
- **Coin Change 2** – https://leetcode.com/problems/coin-change-2/
- **Number of Dice Rolls With Target Sum** – https://leetcode.com/problems/number-of-dice-rolls-with-target-sum/
- **Minimum Subset Sum Difference** – https://leetcode.com/problems/partition-equal-subset-sum/

---

## 🧠 Flashcard (for spaced repetition)

What is the 0/1 Knapsack (DP) pattern? ::

• Classic DP where each item can be included (1) or excluded (0) from knapsack to maximize value
• Uses 2D table or space-optimized 1D array with O(n×W) time complexity
• Applied to subset sum, partition problems, and resource allocation with binary choices

[[01-knapsack-dp]] 