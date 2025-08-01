## 🧠 Permutations (Backtracking)

**Permutations (Backtracking)** generates all possible arrangements of elements using recursive backtracking to systematically place each element.

• Explores all possible orderings of elements (n! permutations for n elements)
• Uses backtracking to place elements and undo choices when exploring alternatives
• Maintains state (visited elements) during recursion
• Time complexity: O(n! × n), Space complexity: O(n) for recursion depth

**Related:** [[Backtracking]] [[Recursion]] [[Array]] [[Combinatorics]]

---

## 💡 Python Code Snippet

```python
def permutations_backtracking(nums):
    """
    Generate all permutations using backtracking
    """
    result = []
    
    def backtrack(current_permutation):
        # Base case: permutation is complete
        if len(current_permutation) == len(nums):
            result.append(current_permutation[:])  # Make a copy
            return
        
        for num in nums:
            # Skip if number is already used
            if num in current_permutation:
                continue
            
            # Include current number
            current_permutation.append(num)
            
            # Recurse to build rest of permutation
            backtrack(current_permutation)
            
            # Backtrack: remove current number
            current_permutation.pop()
    
    backtrack([])
    return result

def permutations_with_visited(nums):
    """
    Generate permutations using visited array for efficiency
    """
    result = []
    visited = [False] * len(nums)
    
    def backtrack(current_permutation):
        if len(current_permutation) == len(nums):
            result.append(current_permutation[:])
            return
        
        for i in range(len(nums)):
            if visited[i]:
                continue
            
            # Mark as visited
            visited[i] = True
            current_permutation.append(nums[i])
            
            backtrack(current_permutation)
            
            # Backtrack
            current_permutation.pop()
            visited[i] = False
    
    backtrack([])
    return result

def permutations_with_duplicates(nums):
    """
    Generate unique permutations from array with duplicate elements
    """
    result = []
    nums.sort()  # Sort to handle duplicates
    visited = [False] * len(nums)
    
    def backtrack(current_permutation):
        if len(current_permutation) == len(nums):
            result.append(current_permutation[:])
            return
        
        for i in range(len(nums)):
            if visited[i]:
                continue
            
            # Skip duplicates: if current element equals previous and 
            # previous element is not used in current permutation
            if i > 0 and nums[i] == nums[i - 1] and not visited[i - 1]:
                continue
            
            visited[i] = True
            current_permutation.append(nums[i])
            
            backtrack(current_permutation)
            
            current_permutation.pop()
            visited[i] = False
    
    backtrack([])
    return result

def next_permutation(nums):
    """
    Generate lexicographically next permutation in-place
    """
    # Find the largest index i such that nums[i] < nums[i + 1]
    i = len(nums) - 2
    while i >= 0 and nums[i] >= nums[i + 1]:
        i -= 1
    
    if i == -1:
        # No next permutation, reverse to get smallest
        nums.reverse()
        return
    
    # Find the largest index j such that nums[i] < nums[j]
    j = len(nums) - 1
    while nums[j] <= nums[i]:
        j -= 1
    
    # Swap nums[i] and nums[j]
    nums[i], nums[j] = nums[j], nums[i]
    
    # Reverse the suffix starting at nums[i + 1]
    nums[i + 1:] = reversed(nums[i + 1:])

def permutation_sequence(n, k):
    """
    Find kth permutation of numbers 1 to n
    """
    import math
    
    # Generate list of numbers
    numbers = list(range(1, n + 1))
    result = []
    k -= 1  # Convert to 0-based indexing
    
    for i in range(n):
        # Calculate factorial for remaining numbers
        factorial = math.factorial(n - 1 - i)
        
        # Find index of number to place at current position
        index = k // factorial
        
        # Add number to result and remove from available numbers
        result.append(numbers.pop(index))
        
        # Update k for next iteration
        k %= factorial
    
    return ''.join(map(str, result))

def letter_permutations(s):
    """
    Generate all permutations of characters in string
    """
    result = []
    chars = list(s)
    
    def backtrack(current_permutation):
        if len(current_permutation) == len(chars):
            result.append(''.join(current_permutation))
            return
        
        used = set()  # Track used characters at current level
        
        for i, char in enumerate(chars):
            if i < len(current_permutation) or char in used:
                continue
            
            used.add(char)
            current_permutation.append(char)
            chars.pop(i)
            
            backtrack(current_permutation)
            
            chars.insert(i, char)
            current_permutation.pop()
    
    backtrack([])
    return result

def permute_unique_optimized(nums):
    """
    Optimized version for unique permutations with duplicates
    """
    from collections import Counter
    
    counter = Counter(nums)
    result = []
    
    def backtrack(current_permutation):
        if len(current_permutation) == len(nums):
            result.append(current_permutation[:])
            return
        
        for num in counter:
            if counter[num] > 0:
                # Use one instance of this number
                counter[num] -= 1
                current_permutation.append(num)
                
                backtrack(current_permutation)
                
                # Backtrack
                current_permutation.pop()
                counter[num] += 1
    
    backtrack([])
    return result

def palindrome_permutations(s):
    """
    Generate all palindromic permutations of string
    """
    from collections import Counter
    
    counter = Counter(s)
    odd_chars = [char for char, count in counter.items() if count % 2 == 1]
    
    # For palindrome, at most one character can have odd count
    if len(odd_chars) > 1:
        return []
    
    # Build first half of palindrome
    first_half = []
    middle = ""
    
    for char, count in counter.items():
        first_half.extend([char] * (count // 2))
        if count % 2 == 1:
            middle = char
    
    # Generate permutations of first half
    result = []
    visited = [False] * len(first_half)
    
    def backtrack(current):
        if len(current) == len(first_half):
            # Create palindrome
            palindrome = ''.join(current) + middle + ''.join(current[::-1])
            result.append(palindrome)
            return
        
        used = set()
        for i in range(len(first_half)):
            if visited[i] or first_half[i] in used:
                continue
            
            used.add(first_half[i])
            visited[i] = True
            current.append(first_half[i])
            
            backtrack(current)
            
            current.pop()
            visited[i] = False
    
    backtrack([])
    return result

def restore_ip_addresses(s):
    """
    Generate all possible valid IP addresses from string
    """
    result = []
    
    def is_valid(segment):
        # Check if segment is valid IP part
        if len(segment) > 3 or len(segment) == 0:
            return False
        if len(segment) > 1 and segment[0] == '0':
            return False
        return 0 <= int(segment) <= 255
    
    def backtrack(start, path):
        # Base case: 4 segments formed
        if len(path) == 4:
            if start == len(s):
                result.append('.'.join(path))
            return
        
        # Try different segment lengths
        for length in range(1, 4):
            if start + length > len(s):
                break
            
            segment = s[start:start + length]
            if is_valid(segment):
                path.append(segment)
                backtrack(start + length, path)
                path.pop()
    
    backtrack(0, [])
    return result

def beautiful_arrangements(n):
    """
    Count beautiful arrangements where arr[i] % i == 0 or i % arr[i] == 0
    """
    def backtrack(index, visited):
        if index > n:
            return 1
        
        count = 0
        for num in range(1, n + 1):
            if not visited[num] and (num % index == 0 or index % num == 0):
                visited[num] = True
                count += backtrack(index + 1, visited)
                visited[num] = False
        
        return count
    
    visited = [False] * (n + 1)
    return backtrack(1, visited)
```

---

## 🔗 LeetCode Practice Problems (10)

- **Permutations** – https://leetcode.com/problems/permutations/
- **Permutations II** – https://leetcode.com/problems/permutations-ii/
- **Next Permutation** – https://leetcode.com/problems/next-permutation/
- **Permutation Sequence** – https://leetcode.com/problems/permutation-sequence/
- **Letter Case Permutation** – https://leetcode.com/problems/letter-case-permutation/
- **Palindrome Permutation II** – https://leetcode.com/problems/palindrome-permutation-ii/
- **Restore IP Addresses** – https://leetcode.com/problems/restore-ip-addresses/
- **Beautiful Arrangement** – https://leetcode.com/problems/beautiful-arrangement/
- **Permutation in String** – https://leetcode.com/problems/permutation-in-string/
- **Find All Anagrams in a String** – https://leetcode.com/problems/find-all-anagrams-in-a-string/

---

## 🧠 Flashcard (for spaced repetition)

What is the Permutations (Backtracking) pattern? ::

• Uses recursive backtracking to generate all possible arrangements (n! permutations) of elements
• Systematically places each element and backtracks to explore alternatives with O(n! × n) complexity
• Applied to arrangement problems, sequence generation, and ordering-dependent solutions

[[permutations-backtracking]] 