## 🧠 Subsets (Backtracking)

**Subsets (Backtracking)** generates all possible subsets of a given set using recursive backtracking to explore include/exclude decisions.

• Uses recursion to make binary decisions: include or exclude each element
• Generates 2^n subsets for n elements (power set)
• Backtracking allows systematic exploration of all possibilities
• Time complexity: O(2^n × n), Space complexity: O(n) for recursion depth

**Related:** [[Backtracking]] [[Recursion]] [[Array]] [[Combinatorics]]

---

## 💡 Python Code Snippet

```python
def subsets_backtracking(nums):
    """
    Generate all subsets using backtracking
    """
    result = []
    
    def backtrack(start, current_subset):
        # Add current subset to result
        result.append(current_subset[:])  # Make a copy
        
        # Explore further elements
        for i in range(start, len(nums)):
            # Include current element
            current_subset.append(nums[i])
            
            # Recurse with next index
            backtrack(i + 1, current_subset)
            
            # Backtrack: remove current element
            current_subset.pop()
    
    backtrack(0, [])
    return result

def subsets_iterative(nums):
    """
    Generate all subsets iteratively using bit manipulation
    """
    result = []
    n = len(nums)
    
    # Generate all possible combinations (2^n)
    for i in range(1 << n):  # 2^n combinations
        subset = []
        for j in range(n):
            # Check if j-th bit is set
            if i & (1 << j):
                subset.append(nums[j])
        result.append(subset)
    
    return result

def subsets_with_duplicates(nums):
    """
    Generate subsets from array with duplicate elements
    """
    result = []
    nums.sort()  # Sort to handle duplicates
    
    def backtrack(start, current_subset):
        result.append(current_subset[:])
        
        for i in range(start, len(nums)):
            # Skip duplicates: if current element equals previous and 
            # previous wasn't included in current path
            if i > start and nums[i] == nums[i - 1]:
                continue
            
            current_subset.append(nums[i])
            backtrack(i + 1, current_subset)
            current_subset.pop()
    
    backtrack(0, [])
    return result

def subsets_of_size_k(nums, k):
    """
    Generate all subsets of specific size k
    """
    result = []
    
    def backtrack(start, current_subset):
        # If subset has k elements, add to result
        if len(current_subset) == k:
            result.append(current_subset[:])
            return
        
        # Early termination: not enough elements left
        remaining = len(nums) - start
        needed = k - len(current_subset)
        if remaining < needed:
            return
        
        for i in range(start, len(nums)):
            current_subset.append(nums[i])
            backtrack(i + 1, current_subset)
            current_subset.pop()
    
    backtrack(0, [])
    return result

def letter_case_permutations(s):
    """
    Generate all possible letter case permutations
    """
    result = []
    
    def backtrack(index, current):
        if index == len(s):
            result.append(current)
            return
        
        char = s[index]
        
        if char.isalpha():
            # Include lowercase version
            backtrack(index + 1, current + char.lower())
            # Include uppercase version
            backtrack(index + 1, current + char.upper())
        else:
            # Digit: include as is
            backtrack(index + 1, current + char)
    
    backtrack(0, "")
    return result

def generate_parentheses(n):
    """
    Generate all valid combinations of n pairs of parentheses
    """
    result = []
    
    def backtrack(current, open_count, close_count):
        # Base case: used all n pairs
        if open_count == n and close_count == n:
            result.append(current)
            return
        
        # Add opening bracket if we haven't used all
        if open_count < n:
            backtrack(current + "(", open_count + 1, close_count)
        
        # Add closing bracket if it doesn't exceed opening brackets
        if close_count < open_count:
            backtrack(current + ")", open_count, close_count + 1)
    
    backtrack("", 0, 0)
    return result

def combination_sum(candidates, target):
    """
    Find all combinations that sum to target (numbers can be reused)
    """
    result = []
    candidates.sort()  # Sort for optimization
    
    def backtrack(start, current_combination, current_sum):
        if current_sum == target:
            result.append(current_combination[:])
            return
        
        if current_sum > target:
            return  # Pruning
        
        for i in range(start, len(candidates)):
            current_combination.append(candidates[i])
            # Can reuse same number, so pass i (not i + 1)
            backtrack(i, current_combination, current_sum + candidates[i])
            current_combination.pop()
    
    backtrack(0, [], 0)
    return result

def combination_sum_ii(candidates, target):
    """
    Find combinations that sum to target (each number used once)
    """
    result = []
    candidates.sort()
    
    def backtrack(start, current_combination, current_sum):
        if current_sum == target:
            result.append(current_combination[:])
            return
        
        if current_sum > target:
            return
        
        for i in range(start, len(candidates)):
            # Skip duplicates
            if i > start and candidates[i] == candidates[i - 1]:
                continue
            
            current_combination.append(candidates[i])
            backtrack(i + 1, current_combination, current_sum + candidates[i])
            current_combination.pop()
    
    backtrack(0, [], 0)
    return result

def word_search_all_paths(board, word):
    """
    Find all paths where word can be formed in the board
    """
    if not board or not board[0]:
        return []
    
    rows, cols = len(board), len(board[0])
    result = []
    
    def backtrack(row, col, path, word_index):
        # Found complete word
        if word_index == len(word):
            result.append(path[:])
            return
        
        # Check bounds
        if (row < 0 or row >= rows or col < 0 or col >= cols or 
            board[row][col] != word[word_index] or (row, col) in path):
            return
        
        # Mark current cell as visited
        path.add((row, col))
        
        # Explore all 4 directions
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for dr, dc in directions:
            backtrack(row + dr, col + dc, path, word_index + 1)
        
        # Backtrack: unmark current cell
        path.remove((row, col))
    
    # Try starting from each cell
    for i in range(rows):
        for j in range(cols):
            if board[i][j] == word[0]:
                backtrack(i, j, set(), 0)
    
    return result

def palindrome_partitioning(s):
    """
    Find all palindromic partitions of string
    """
    result = []
    
    def is_palindrome(string):
        return string == string[::-1]
    
    def backtrack(start, current_partition):
        if start == len(s):
            result.append(current_partition[:])
            return
        
        for end in range(start + 1, len(s) + 1):
            substring = s[start:end]
            if is_palindrome(substring):
                current_partition.append(substring)
                backtrack(end, current_partition)
                current_partition.pop()
    
    backtrack(0, [])
    return result

def binary_watch(turnedOn):
    """
    Generate all possible times with given number of LEDs turned on
    """
    result = []
    
    def count_bits(num):
        count = 0
        while num:
            count += 1
            num &= num - 1  # Remove rightmost set bit
        return count
    
    # Hours: 0-11 (4 bits), Minutes: 0-59 (6 bits)
    for h in range(12):
        for m in range(60):
            if count_bits(h) + count_bits(m) == turnedOn:
                result.append(f"{h}:{m:02d}")
    
    return result
```

---

## 🔗 LeetCode Practice Problems (10)

- **Subsets** – https://leetcode.com/problems/subsets/
- **Subsets II** – https://leetcode.com/problems/subsets-ii/
- **Letter Case Permutation** – https://leetcode.com/problems/letter-case-permutation/
- **Generate Parentheses** – https://leetcode.com/problems/generate-parentheses/
- **Combination Sum** – https://leetcode.com/problems/combination-sum/
- **Combination Sum II** – https://leetcode.com/problems/combination-sum-ii/
- **Palindrome Partitioning** – https://leetcode.com/problems/palindrome-partitioning/
- **Word Search** – https://leetcode.com/problems/word-search/
- **Binary Watch** – https://leetcode.com/problems/binary-watch/
- **Combinations** – https://leetcode.com/problems/combinations/

---

## 🧠 Flashcard (for spaced repetition)

What is the Subsets (Backtracking) pattern? :: • Uses recursive backtracking to generate all possible subsets by making include/exclude decisions • Explores 2^n possibilities systematically with O(2^n × n) time complexity • Applied to power set generation, combinations, and problems requiring all possible selections [[subsets-backtracking]] 