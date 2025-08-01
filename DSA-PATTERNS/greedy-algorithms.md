## 🧠 Greedy Algorithms

**Greedy Algorithms** make locally optimal choices at each step with the hope of achieving a globally optimal solution.

• Makes the best immediate choice without considering future consequences
• Works when local optimal choices lead to global optimal solution
• Generally easier to implement than dynamic programming but harder to prove correctness
• Common applications: scheduling, interval problems, and optimization tasks

**Related:** [[Optimization]] [[Sorting]] [[Scheduling]]

---

## 💡 Python Code Snippet

```python
def activity_selection(activities):
    """
    Select maximum number of non-overlapping activities
    Activities: [(start_time, end_time), ...]
    """
    # Sort by end time (greedy choice: pick activity ending earliest)
    activities.sort(key=lambda x: x[1])
    
    selected = [activities[0]]
    last_end_time = activities[0][1]
    
    for start, end in activities[1:]:
        # If activity doesn't overlap with last selected
        if start >= last_end_time:
            selected.append((start, end))
            last_end_time = end
    
    return selected

def fractional_knapsack(items, capacity):
    """
    Fractional knapsack: items can be broken into fractions
    Items: [(value, weight), ...]
    """
    # Sort by value-to-weight ratio (greedy choice)
    items.sort(key=lambda x: x[0]/x[1], reverse=True)
    
    total_value = 0
    remaining_capacity = capacity
    
    for value, weight in items:
        if weight <= remaining_capacity:
            # Take entire item
            total_value += value
            remaining_capacity -= weight
        else:
            # Take fraction of item
            fraction = remaining_capacity / weight
            total_value += value * fraction
            break
    
    return total_value

def minimum_coins_greedy(coins, amount):
    """
    Minimum coins to make amount using greedy approach
    Note: Only works for canonical coin systems (like US coins)
    """
    coins.sort(reverse=True)  # Start with largest denomination
    
    coin_count = 0
    for coin in coins:
        if amount >= coin:
            count = amount // coin
            coin_count += count
            amount -= count * coin
            
            if amount == 0:
                break
    
    return coin_count if amount == 0 else -1

def gas_station_circuit(gas, cost):
    """
    Find starting gas station to complete circuit
    """
    total_tank = 0
    current_tank = 0
    starting_station = 0
    
    for i in range(len(gas)):
        total_tank += gas[i] - cost[i]
        current_tank += gas[i] - cost[i]
        
        # If current tank becomes negative, reset starting point
        if current_tank < 0:
            starting_station = i + 1
            current_tank = 0
    
    return starting_station if total_tank >= 0 else -1

def jump_game_greedy(nums):
    """
    Check if we can reach the last index
    """
    max_reach = 0
    
    for i in range(len(nums)):
        # If current position is beyond max reachable, return False
        if i > max_reach:
            return False
        
        # Update max reachable position
        max_reach = max(max_reach, i + nums[i])
        
        # If we can reach or exceed last index
        if max_reach >= len(nums) - 1:
            return True
    
    return False

def minimum_jumps(nums):
    """
    Find minimum number of jumps to reach end
    """
    if len(nums) <= 1:
        return 0
    
    jumps = 0
    current_max = 0  # Farthest we can reach with current jumps
    next_max = 0     # Farthest we can reach with one more jump
    
    for i in range(len(nums) - 1):
        next_max = max(next_max, i + nums[i])
        
        # If we've reached the limit of current jump range
        if i == current_max:
            jumps += 1
            current_max = next_max
            
            # If we can reach the end
            if current_max >= len(nums) - 1:
                break
    
    return jumps

def meeting_rooms_ii(intervals):
    """
    Minimum meeting rooms required
    """
    if not intervals:
        return 0
    
    # Separate start and end times
    starts = sorted([interval[0] for interval in intervals])
    ends = sorted([interval[1] for interval in intervals])
    
    rooms_needed = 0
    max_rooms = 0
    start_ptr = end_ptr = 0
    
    # Greedy approach: process events in chronological order
    while start_ptr < len(starts):
        if starts[start_ptr] < ends[end_ptr]:
            # Meeting starts, need a room
            rooms_needed += 1
            start_ptr += 1
        else:
            # Meeting ends, free a room
            rooms_needed -= 1
            end_ptr += 1
        
        max_rooms = max(max_rooms, rooms_needed)
    
    return max_rooms

def remove_k_digits(num, k):
    """
    Remove k digits to form smallest possible number
    """
    stack = []
    
    for digit in num:
        # Remove larger digits from stack (greedy choice)
        while stack and stack[-1] > digit and k > 0:
            stack.pop()
            k -= 1
        
        stack.append(digit)
    
    # Remove remaining digits from end if k > 0
    while k > 0:
        stack.pop()
        k -= 1
    
    # Build result, handle leading zeros
    result = ''.join(stack).lstrip('0')
    return result if result else '0'

def candy_distribution(ratings):
    """
    Distribute candies such that higher rated children get more candy
    """
    n = len(ratings)
    candies = [1] * n
    
    # Left to right pass: ensure right neighbor with higher rating gets more
    for i in range(1, n):
        if ratings[i] > ratings[i-1]:
            candies[i] = candies[i-1] + 1
    
    # Right to left pass: ensure left neighbor with higher rating gets more
    for i in range(n-2, -1, -1):
        if ratings[i] > ratings[i+1]:
            candies[i] = max(candies[i], candies[i+1] + 1)
    
    return sum(candies)

def non_overlapping_intervals(intervals):
    """
    Minimum number of intervals to remove to make rest non-overlapping
    """
    if not intervals:
        return 0
    
    # Sort by end time (greedy: keep intervals ending earliest)
    intervals.sort(key=lambda x: x[1])
    
    count = 0
    end = intervals[0][1]
    
    for i in range(1, len(intervals)):
        if intervals[i][0] < end:
            # Overlapping interval, remove it
            count += 1
        else:
            # Non-overlapping, update end time
            end = intervals[i][1]
    
    return count

def partition_labels(s):
    """
    Partition string into as many parts as possible where each letter appears in at most one part
    """
    # Find last occurrence of each character
    last_occurrence = {char: i for i, char in enumerate(s)}
    
    partitions = []
    start = 0
    max_reach = 0
    
    for i, char in enumerate(s):
        max_reach = max(max_reach, last_occurrence[char])
        
        # If we've reached the farthest any character in current partition can go
        if i == max_reach:
            partitions.append(i - start + 1)
            start = i + 1
    
    return partitions

def task_scheduler_greedy(tasks, n):
    """
    Minimum time to execute all tasks with cooldown period n
    """
    from collections import Counter
    
    # Count frequency of each task
    task_counts = Counter(tasks)
    max_freq = max(task_counts.values())
    
    # Count how many tasks have maximum frequency
    max_freq_count = sum(1 for count in task_counts.values() if count == max_freq)
    
    # Minimum time needed
    # Either total tasks or time needed to space out most frequent tasks
    return max(len(tasks), (max_freq - 1) * (n + 1) + max_freq_count)

def boats_to_save_people(people, limit):
    """
    Minimum boats needed to rescue people (max 2 people per boat)
    """
    people.sort()
    left, right = 0, len(people) - 1
    boats = 0
    
    while left <= right:
        # Always take the heaviest person
        if left == right:
            # Only one person left
            boats += 1
            break
        
        # Try to pair lightest with heaviest
        if people[left] + people[right] <= limit:
            left += 1  # Take lightest person too
        
        right -= 1  # Always take heaviest person
        boats += 1
    
    return boats

def maximum_swap(num):
    """
    Maximum number you can get by swapping two digits at most once
    """
    digits = list(str(num))
    n = len(digits)
    
    # Find rightmost occurrence of each digit
    last = {}
    for i, digit in enumerate(digits):
        last[digit] = i
    
    # Try to find the first digit that can be improved
    for i in range(n):
        # Look for a larger digit that appears later
        for d in range(9, int(digits[i]), -1):
            if str(d) in last and last[str(d)] > i:
                # Swap with the rightmost occurrence of larger digit
                j = last[str(d)]
                digits[i], digits[j] = digits[j], digits[i]
                return int(''.join(digits))
    
    return num  # No beneficial swap found
```

---

## 🔗 LeetCode Practice Problems (10)

- **Jump Game** – https://leetcode.com/problems/jump-game/
- **Jump Game II** – https://leetcode.com/problems/jump-game-ii/
- **Gas Station** – https://leetcode.com/problems/gas-station/
- **Candy** – https://leetcode.com/problems/candy/
- **Non-overlapping Intervals** – https://leetcode.com/problems/non-overlapping-intervals/
- **Meeting Rooms II** – https://leetcode.com/problems/meeting-rooms-ii/
- **Partition Labels** – https://leetcode.com/problems/partition-labels/
- **Task Scheduler** – https://leetcode.com/problems/task-scheduler/
- **Remove K Digits** – https://leetcode.com/problems/remove-k-digits/
- **Boats to Save People** – https://leetcode.com/problems/boats-to-save-people/

---

## 🧠 Flashcard (for spaced repetition)

What is the Greedy Algorithms pattern? ::

• Makes locally optimal choices at each step hoping to achieve globally optimal solution
• Works when local optimal choices lead to global optimum, often requiring proof of correctness
• Applied to scheduling, interval problems, optimization tasks, and resource allocation problems

[[greedy-algorithms]] 