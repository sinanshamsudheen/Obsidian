## 🧠 Unbounded Knapsack (DP)

**Unbounded Knapsack** allows using each item unlimited times, unlike 0/1 knapsack where each item can be used at most once.

• Items can be selected multiple times (unlimited supply)
• Goal is to maximize value within weight/capacity constraints  
• Uses 1D DP array, processing in forward direction
• Time complexity: O(n × W), Space: O(W)

**Related:** [[Dynamic Programming]] [[Array]] [[Optimization]]

---

## 💡 Python Code Snippet

```python
def unbounded_knapsack(weights, values, capacity):
    """
    Unbounded knapsack - each item can be used multiple times
    """
    dp = [0] * (capacity + 1)
    
    # For each capacity
    for w in range(1, capacity + 1):
        # Try each item
        for i in range(len(weights)):
            if weights[i] <= w:
                dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    
    return dp[capacity]

def unbounded_knapsack_with_items(weights, values, capacity):
    """
    Unbounded knapsack that also returns the items used
    """
    dp = [0] * (capacity + 1)
    items_used = [[] for _ in range(capacity + 1)]
    
    for w in range(1, capacity + 1):
        for i in range(len(weights)):
            if weights[i] <= w:
                new_value = dp[w - weights[i]] + values[i]
                if new_value > dp[w]:
                    dp[w] = new_value
                    items_used[w] = items_used[w - weights[i]] + [i]
    
    return dp[capacity], items_used[capacity]

def coin_change_min_coins(coins, amount):
    """
    Minimum number of coins to make amount (unbounded)
    """
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1

def coin_change_ways(coins, amount):
    """
    Number of ways to make amount using coins (each coin unlimited)
    """
    dp = [0] * (amount + 1)
    dp[0] = 1  # One way to make 0: use no coins
    
    # Process each coin type
    for coin in coins:
        # Update dp for all amounts that can use this coin
        for i in range(coin, amount + 1):
            dp[i] += dp[i - coin]
    
    return dp[amount]

def perfect_squares(n):
    """
    Minimum number of perfect squares that sum to n
    """
    # Generate perfect squares up to n
    squares = []
    i = 1
    while i * i <= n:
        squares.append(i * i)
        i += 1
    
    dp = [float('inf')] * (n + 1)
    dp[0] = 0
    
    for i in range(1, n + 1):
        for square in squares:
            if square <= i:
                dp[i] = min(dp[i], dp[i - square] + 1)
    
    return dp[n]

def combination_sum_iv(nums, target):
    """
    Number of combinations that add up to target (order matters)
    """
    dp = [0] * (target + 1)
    dp[0] = 1
    
    # For each target value
    for i in range(1, target + 1):
        # Try each number
        for num in nums:
            if num <= i:
                dp[i] += dp[i - num]
    
    return dp[target]

def rod_cutting(prices, length):
    """
    Maximum revenue from cutting rod of given length
    prices[i] = price of rod of length i+1
    """
    dp = [0] * (length + 1)
    
    for i in range(1, length + 1):
        for j in range(i):
            # Cut of length j+1 costs prices[j]
            dp[i] = max(dp[i], dp[i - j - 1] + prices[j])
    
    return dp[length]

def min_cost_climbing_stairs(cost):
    """
    Minimum cost to reach top of stairs (can climb 1 or 2 steps)
    """
    n = len(cost)
    dp = [0] * (n + 1)
    
    for i in range(2, n + 1):
        dp[i] = min(dp[i-1] + cost[i-1], dp[i-2] + cost[i-2])
    
    return dp[n]

def decode_ways(s):
    """
    Number of ways to decode string where '1'-'26' map to 'A'-'Z'
    """
    if not s or s[0] == '0':
        return 0
    
    n = len(s)
    dp = [0] * (n + 1)
    dp[0] = 1  # Empty string
    dp[1] = 1  # First character
    
    for i in range(2, n + 1):
        # Single digit
        if s[i-1] != '0':
            dp[i] += dp[i-1]
        
        # Two digits
        two_digit = int(s[i-2:i])
        if 10 <= two_digit <= 26:
            dp[i] += dp[i-2]
    
    return dp[n]

def word_break(s, word_dict):
    """
    Check if string can be segmented into dictionary words
    """
    n = len(s)
    dp = [False] * (n + 1)
    dp[0] = True  # Empty string
    
    word_set = set(word_dict)
    
    for i in range(1, n + 1):
        for j in range(i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break
    
    return dp[n]

def house_robber_circular(nums):
    """
    Maximum money robbed from circular houses (can't rob adjacent)
    """
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    
    def rob_linear(houses):
        prev2 = prev1 = 0
        for money in houses:
            current = max(prev1, prev2 + money)
            prev2 = prev1
            prev1 = current
        return prev1
    
    # Case 1: Rob first house, can't rob last
    case1 = rob_linear(nums[:-1])
    
    # Case 2: Don't rob first house, can rob last
    case2 = rob_linear(nums[1:])
    
    return max(case1, case2)

def delete_and_earn(nums):
    """
    Maximum points earned by deleting numbers (deleting x removes all x-1 and x+1)
    """
    if not nums:
        return 0
    
    # Count frequency and find max
    from collections import Counter
    count = Counter(nums)
    max_num = max(nums)
    
    # DP: earn[i] = max points using numbers up to i
    earn = [0] * (max_num + 1)
    
    for i in range(1, max_num + 1):
        # Don't take i: earn[i-1]
        # Take i: earn[i-2] + i * count[i]
        earn[i] = max(earn[i-1], earn[i-2] + i * count[i])
    
    return earn[max_num]
```

---

## 🔗 LeetCode Practice Problems (10)

- **Coin Change** – https://leetcode.com/problems/coin-change/
- **Coin Change 2** – https://leetcode.com/problems/coin-change-2/
- **Perfect Squares** – https://leetcode.com/problems/perfect-squares/
- **Combination Sum IV** – https://leetcode.com/problems/combination-sum-iv/
- **Min Cost Climbing Stairs** – https://leetcode.com/problems/min-cost-climbing-stairs/
- **Decode Ways** – https://leetcode.com/problems/decode-ways/
- **Word Break** – https://leetcode.com/problems/word-break/
- **House Robber II** – https://leetcode.com/problems/house-robber-ii/
- **Delete and Earn** – https://leetcode.com/problems/delete-and-earn/
- **Integer Break** – https://leetcode.com/problems/integer-break/

---

## 🧠 Flashcard (for spaced repetition)

What is the Unbounded Knapsack (DP) pattern? :: • Items can be selected multiple times (unlimited supply) to maximize value within constraints • Uses 1D DP array processing in forward direction with O(n×W) time complexity • Applied to coin change, perfect squares, and problems allowing repeated use of elements [[unbounded-knapsack-dp]] 