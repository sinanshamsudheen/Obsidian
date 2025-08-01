## 🧠 Palindromic Substrings (DP)

**Palindromic Substrings** pattern finds palindromes within strings using dynamic programming or expand-around-center techniques.

• A palindrome reads the same forwards and backwards
• Can find longest palindromic substring, count all palindromes, or check palindromic subsequences
• Two main approaches: 2D DP table or expand around centers
• Time complexity: O(n²) for most variants

**Related:** [[Dynamic Programming]] [[String]] [[Two Pointers]]

---

## 💡 Python Code Snippet

```python
def longest_palindromic_substring_dp(s):
    """
    Find longest palindromic substring using 2D DP
    dp[i][j] = True if s[i:j+1] is palindrome
    """
    if not s:
        return ""
    
    n = len(s)
    dp = [[False] * n for _ in range(n)]
    start = 0
    max_length = 1
    
    # Every single character is palindrome
    for i in range(n):
        dp[i][i] = True
    
    # Check for palindromes of length 2
    for i in range(n - 1):
        if s[i] == s[i + 1]:
            dp[i][i + 1] = True
            start = i
            max_length = 2
    
    # Check for palindromes of length 3 and more
    for length in range(3, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            
            # Check if current substring is palindrome
            if s[i] == s[j] and dp[i + 1][j - 1]:
                dp[i][j] = True
                start = i
                max_length = length
    
    return s[start:start + max_length]

def longest_palindromic_substring_expand(s):
    """
    Find longest palindromic substring by expanding around centers
    """
    if not s:
        return ""
    
    def expand_around_center(left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return right - left - 1
    
    start = 0
    max_length = 0
    
    for i in range(len(s)):
        # Odd length palindromes (center at i)
        len1 = expand_around_center(i, i)
        
        # Even length palindromes (center between i and i+1)
        len2 = expand_around_center(i, i + 1)
        
        current_max = max(len1, len2)
        
        if current_max > max_length:
            max_length = current_max
            start = i - (current_max - 1) // 2
    
    return s[start:start + max_length]

def count_palindromic_substrings(s):
    """
    Count all palindromic substrings in string
    """
    if not s:
        return 0
    
    def expand_around_center(left, right):
        count = 0
        while left >= 0 and right < len(s) and s[left] == s[right]:
            count += 1
            left -= 1
            right += 1
        return count
    
    total_count = 0
    
    for i in range(len(s)):
        # Odd length palindromes
        total_count += expand_around_center(i, i)
        
        # Even length palindromes
        total_count += expand_around_center(i, i + 1)
    
    return total_count

def longest_palindromic_subsequence(s):
    """
    Find length of longest palindromic subsequence using DP
    """
    n = len(s)
    dp = [[0] * n for _ in range(n)]
    
    # Every character is palindrome of length 1
    for i in range(n):
        dp[i][i] = 1
    
    # Check for palindromes of length 2 and more
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            
            if s[i] == s[j]:
                if length == 2:
                    dp[i][j] = 2
                else:
                    dp[i][j] = dp[i + 1][j - 1] + 2
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
    
    return dp[0][n - 1]

def min_insertions_palindrome(s):
    """
    Minimum insertions to make string palindrome
    """
    lps_length = longest_palindromic_subsequence(s)
    return len(s) - lps_length

def palindromic_partitioning_dp(s):
    """
    Minimum cuts needed to partition string into palindromes
    """
    n = len(s)
    
    # is_palindrome[i][j] = True if s[i:j+1] is palindrome
    is_palindrome = [[False] * n for _ in range(n)]
    
    # Fill palindrome table
    for i in range(n):
        is_palindrome[i][i] = True
    
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j]:
                if length == 2:
                    is_palindrome[i][j] = True
                else:
                    is_palindrome[i][j] = is_palindrome[i + 1][j - 1]
    
    # dp[i] = minimum cuts for s[0:i+1]
    dp = [0] * n
    
    for i in range(n):
        if is_palindrome[0][i]:
            dp[i] = 0
        else:
            dp[i] = float('inf')
            for j in range(i):
                if is_palindrome[j + 1][i]:
                    dp[i] = min(dp[i], dp[j] + 1)
    
    return dp[n - 1]

def all_palindromic_partitions(s):
    """
    Find all possible palindromic partitions
    """
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
    
    result = []
    backtrack(0, [])
    return result

def valid_palindrome_ii(s):
    """
    Check if string can be palindrome after deleting at most one character
    """
    def is_palindrome(left, right):
        while left < right:
            if s[left] != s[right]:
                return False
            left += 1
            right -= 1
        return True
    
    left, right = 0, len(s) - 1
    
    while left < right:
        if s[left] != s[right]:
            # Try deleting either left or right character
            return is_palindrome(left + 1, right) or is_palindrome(left, right - 1)
        left += 1
        right -= 1
    
    return True

def shortest_palindrome(s):
    """
    Find shortest palindrome by adding characters at beginning
    """
    if not s:
        return s
    
    # Find longest palindromic prefix
    def get_longest_palindromic_prefix():
        combined = s + "#" + s[::-1]
        n = len(combined)
        lps = [0] * n  # Longest Proper Prefix which is also Suffix
        
        for i in range(1, n):
            j = lps[i - 1]
            while j > 0 and combined[i] != combined[j]:
                j = lps[j - 1]
            if combined[i] == combined[j]:
                j += 1
            lps[i] = j
        
        return lps[n - 1]
    
    prefix_length = get_longest_palindromic_prefix()
    suffix_to_add = s[prefix_length:][::-1]
    return suffix_to_add + s

def count_different_palindromic_subsequences(s):
    """
    Count distinct palindromic subsequences
    """
    n = len(s)
    MOD = 10**9 + 7
    
    # dp[i][j] = number of distinct palindromic subsequences in s[i:j+1]
    dp = [[0] * n for _ in range(n)]
    
    # Single characters
    for i in range(n):
        dp[i][i] = 1
    
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            
            if s[i] == s[j]:
                low, high = i + 1, j - 1
                
                # Find first and last occurrence of s[i] in s[i+1:j]
                while low <= high and s[low] != s[i]:
                    low += 1
                while low <= high and s[high] != s[i]:
                    high -= 1
                
                if low > high:
                    # No occurrence of s[i] in between
                    dp[i][j] = (dp[i + 1][j - 1] * 2 + 2) % MOD
                elif low == high:
                    # One occurrence
                    dp[i][j] = (dp[i + 1][j - 1] * 2 + 1) % MOD
                else:
                    # Multiple occurrences
                    dp[i][j] = (dp[i + 1][j - 1] * 2 - dp[low + 1][high - 1]) % MOD
            else:
                dp[i][j] = (dp[i + 1][j] + dp[i][j - 1] - dp[i + 1][j - 1]) % MOD
    
    return dp[0][n - 1] % MOD
```

---

## 🔗 LeetCode Practice Problems (10)

- **Longest Palindromic Substring** – https://leetcode.com/problems/longest-palindromic-substring/
- **Palindromic Substrings** – https://leetcode.com/problems/palindromic-substrings/
- **Longest Palindromic Subsequence** – https://leetcode.com/problems/longest-palindromic-subsequence/
- **Palindrome Partitioning** – https://leetcode.com/problems/palindrome-partitioning/
- **Palindrome Partitioning II** – https://leetcode.com/problems/palindrome-partitioning-ii/
- **Valid Palindrome II** – https://leetcode.com/problems/valid-palindrome-ii/
- **Shortest Palindrome** – https://leetcode.com/problems/shortest-palindrome/
- **Count Different Palindromic Subsequences** – https://leetcode.com/problems/count-different-palindromic-subsequences/
- **Minimum Insertion Steps to Make a String Palindrome** – https://leetcode.com/problems/minimum-insertion-steps-to-make-a-string-palindrome/
- **Palindrome Pairs** – https://leetcode.com/problems/palindrome-pairs/

---

## 🧠 Flashcard (for spaced repetition)

What is the Palindromic Substrings (DP) pattern? :: • Finds palindromes (strings reading same forwards/backwards) using DP table or expand-around-centers • Two approaches: 2D DP table O(n²) space or expand-around-center O(1) space • Used for longest palindromic substring, counting palindromes, and palindromic partitioning problems [[palindromic-substrings-dp]] 