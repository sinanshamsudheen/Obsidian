## 🧠 Longest Common Subsequence (DP)

**Longest Common Subsequence (LCS)** finds the longest subsequence common to two or more sequences, where subsequence maintains relative order but elements need not be consecutive.

• Subsequence preserves order but allows gaps (unlike substring)
• Uses 2D DP table where dp[i][j] = LCS length of first i chars of string1 and first j chars of string2
• Time complexity: O(m × n), Space: O(m × n) or O(min(m,n)) optimized
• Foundation for edit distance, diff algorithms, and sequence alignment

**Related:** [[Dynamic Programming]] [[String]] [[Sequence]]

---

## 💡 Python Code Snippet

```python
def longest_common_subsequence(text1, text2):
    """
    Find length of longest common subsequence
    """
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]

def lcs_with_sequence(text1, text2):
    """
    Find the actual longest common subsequence string
    """
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    # Backtrack to find the sequence
    lcs = []
    i, j = m, n
    
    while i > 0 and j > 0:
        if text1[i-1] == text2[j-1]:
            lcs.append(text1[i-1])
            i -= 1
            j -= 1
        elif dp[i-1][j] > dp[i][j-1]:
            i -= 1
        else:
            j -= 1
    
    return ''.join(reversed(lcs))

def lcs_space_optimized(text1, text2):
    """
    Space optimized LCS using O(min(m,n)) space
    """
    # Make text1 the shorter string
    if len(text1) > len(text2):
        text1, text2 = text2, text1
    
    m, n = len(text1), len(text2)
    prev = [0] * (m + 1)
    curr = [0] * (m + 1)
    
    for j in range(1, n + 1):
        for i in range(1, m + 1):
            if text1[i-1] == text2[j-1]:
                curr[i] = prev[i-1] + 1
            else:
                curr[i] = max(prev[i], curr[i-1])
        
        prev, curr = curr, prev
    
    return prev[m]

def longest_common_substring(text1, text2):
    """
    Find length of longest common substring (consecutive characters)
    """
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_length = 0
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
                max_length = max(max_length, dp[i][j])
            else:
                dp[i][j] = 0
    
    return max_length

def edit_distance(word1, word2):
    """
    Minimum edit distance (insertions, deletions, substitutions)
    """
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i  # Delete all characters
    for j in range(n + 1):
        dp[0][j] = j  # Insert all characters
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]  # No operation needed
            else:
                dp[i][j] = min(
                    dp[i-1][j] + 1,    # Deletion
                    dp[i][j-1] + 1,    # Insertion
                    dp[i-1][j-1] + 1   # Substitution
                )
    
    return dp[m][n]

def longest_palindromic_subsequence(s):
    """
    Length of longest palindromic subsequence (LCS with reverse)
    """
    return longest_common_subsequence(s, s[::-1])

def shortest_common_supersequence_length(str1, str2):
    """
    Length of shortest string that contains both str1 and str2 as subsequences
    """
    lcs_length = longest_common_subsequence(str1, str2)
    return len(str1) + len(str2) - lcs_length

def min_insertions_deletions(word1, word2):
    """
    Minimum insertions and deletions to convert word1 to word2
    """
    lcs_length = longest_common_subsequence(word1, word2)
    deletions = len(word1) - lcs_length
    insertions = len(word2) - lcs_length
    return deletions + insertions

def longest_increasing_subsequence(nums):
    """
    Length of longest increasing subsequence
    """
    if not nums:
        return 0
    
    n = len(nums)
    dp = [1] * n  # dp[i] = LIS length ending at index i
    
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    
    return max(dp)

def lis_binary_search(nums):
    """
    LIS using binary search - O(n log n)
    """
    if not nums:
        return 0
    
    from bisect import bisect_left
    
    tails = []  # tails[i] = smallest tail element for LIS of length i+1
    
    for num in nums:
        pos = bisect_left(tails, num)
        if pos == len(tails):
            tails.append(num)
        else:
            tails[pos] = num
    
    return len(tails)

def max_length_pair_chain(pairs):
    """
    Maximum length of chain where pairs can be chained if b < c for [a,b] and [c,d]
    """
    pairs.sort(key=lambda x: x[1])  # Sort by second element
    count = 0
    current_end = float('-inf')
    
    for pair in pairs:
        if pair[0] > current_end:
            count += 1
            current_end = pair[1]
    
    return count

def distinct_subsequences(s, t):
    """
    Number of distinct subsequences of s that equal t
    """
    m, n = len(s), len(t)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Base case: empty t can be formed by any prefix of s in 1 way
    for i in range(m + 1):
        dp[i][0] = 1
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Don't use current character from s
            dp[i][j] = dp[i-1][j]
            
            # Use current character if it matches
            if s[i-1] == t[j-1]:
                dp[i][j] += dp[i-1][j-1]
    
    return dp[m][n]

def interleaving_string(s1, s2, s3):
    """
    Check if s3 is formed by interleaving s1 and s2
    """
    m, n, l = len(s1), len(s2), len(s3)
    
    if m + n != l:
        return False
    
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True
    
    # Fill first row (using only s2)
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j-1] and s2[j-1] == s3[j-1]
    
    # Fill first column (using only s1)
    for i in range(1, m + 1):
        dp[i][0] = dp[i-1][0] and s1[i-1] == s3[i-1]
    
    # Fill rest of table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = (dp[i-1][j] and s1[i-1] == s3[i+j-1]) or \
                       (dp[i][j-1] and s2[j-1] == s3[i+j-1])
    
    return dp[m][n]
```

---

## 🔗 LeetCode Practice Problems (10)

- **Longest Common Subsequence** – https://leetcode.com/problems/longest-common-subsequence/
- **Edit Distance** – https://leetcode.com/problems/edit-distance/
- **Longest Palindromic Subsequence** – https://leetcode.com/problems/longest-palindromic-subsequence/
- **Longest Increasing Subsequence** – https://leetcode.com/problems/longest-increasing-subsequence/
- **Distinct Subsequences** – https://leetcode.com/problems/distinct-subsequences/
- **Shortest Common Supersequence** – https://leetcode.com/problems/shortest-common-supersequence/
- **Interleaving String** – https://leetcode.com/problems/interleaving-string/
- **Maximum Length of Pair Chain** – https://leetcode.com/problems/maximum-length-of-pair-chain/
- **Delete Operation for Two Strings** – https://leetcode.com/problems/delete-operation-for-two-strings/
- **Minimum ASCII Delete Sum for Two Strings** – https://leetcode.com/problems/minimum-ascii-delete-sum-for-two-strings/

---

## 🧠 Flashcard (for spaced repetition)

What is the Longest Common Subsequence (DP) pattern? ::

• Finds longest subsequence common to sequences, preserving order but allowing gaps  
• Uses 2D DP table with O(m×n) time, comparing characters and taking max of previous states
• Foundation for edit distance, sequence alignment, and string similarity problems

[[longest-common-subsequence-dp]] 