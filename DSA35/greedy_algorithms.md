# Greedy Algorithms

## Explanation

Greedy algorithms make locally optimal choices at each step with the hope of finding a global optimum. This pattern works for problems with optimal substructure where the greedy choice property holds. It's efficient but doesn't work for all problems - only those where local choices lead to global solutions.

## Algorithm (Word-based)

- Identify what to optimize
- Make the locally optimal choice at each step
- Prove that the choice maintains optimality
- Continue until problem is solved
- Verify the approach works for the specific problem

## Code Template

```python
def greedy_template(items, constraints):
    """
    General template for greedy algorithms
    """
    # Sort items by some greedy criteria
    sorted_items = sorted(items, key=lambda x: greedy_criterion(x))
    
    result = []
    current_state = initial_state()
    
    for item in sorted_items:
        if is_valid_choice(item, current_state, constraints):
            # Make the greedy choice
            result.append(item)
            update_state(current_state, item)
    
    return result

def activity_selection(activities):
    """
    Select maximum number of non-overlapping activities
    """
    # Sort activities by finish time (greedy choice)
    activities.sort(key=lambda x: x[1])  # x[1] is finish time
    
    selected = []
    last_finish_time = float('-inf')
    
    for start, finish in activities:
        if start >= last_finish_time:  # No overlap
            selected.append((start, finish))
            last_finish_time = finish
    
    return selected

def coin_change_greedy(coins, amount):
    """
    Make change with minimum coins (only works for canonical coin systems)
    """
    # Sort coins in descending order
    coins.sort(reverse=True)
    
    result = 0
    remaining = amount
    
    for coin in coins:
        if remaining >= coin:
            count = remaining // coin
            result += count
            remaining -= count * coin
    
    return result if remaining == 0 else -1

def fractional_knapsack(items, capacity):
    """
    Fractional knapsack with greedy approach
    """
    # Calculate value-to-weight ratio and sort
    items_with_ratio = [(value/weight, weight, value) for weight, value in items]
    items_with_ratio.sort(reverse=True)  # Sort by ratio in descending order
    
    total_value = 0
    remaining_capacity = capacity
    
    for ratio, weight, value in items_with_ratio:
        if remaining_capacity >= weight:
            # Take the whole item
            total_value += value
            remaining_capacity -= weight
        else:
            # Take fraction of the item
            total_value += ratio * remaining_capacity
            break
    
    return total_value
```

## Practice Questions

1. [Assign Cookies](https://leetcode.com/problems/assign-cookies/) - Easy
2. [Best Time to Buy and Sell Stock II](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/) - Easy
3. [Lemonade Change](https://leetcode.com/problems/lemonade-change/) - Easy
4. [Maximum Product of Three Numbers](https://leetcode.com/problems/maximum-product-of-three-numbers/) - Easy
5. [Partition Labels](https://leetcode.com/problems/partition-labels/) - Medium
6. [Remove K Digits](https://leetcode.com/problems/remove-k-digits/) - Medium
7. [Task Scheduler](https://leetcode.com/problems/task-scheduler/) - Medium
8. [Queue Reconstruction by Height](https://leetcode.com/problems/queue-reconstruction-by-height/) - Medium
9. [Non-overlapping Intervals](https://leetcode.com/problems/non-overlapping-intervals/) - Medium
10. [Minimum Number of Arrows to Burst Balloons](https://leetcode.com/problems/minimum-number-of-arrows-to-burst-balloons/) - Medium

## Notes

- Only works if greedy choice property holds
- Needs proof of correctness for each problem
- Often gives optimal solution faster than DP

---

**Tags:** #dsa #greedy-algorithms #optimization #leetcode

**Last Reviewed:** 2025-11-05