## 🧠 Sliding Window

The **Sliding Window** pattern involves creating a window of fixed or variable size that slides through a data structure (usually an array or string). 

• Instead of using nested loops, we maintain a window and adjust its boundaries
• Useful for problems involving contiguous subarrays or substrings
• Helps reduce time complexity from O(n²) to O(n)
• Two main types: fixed-size window and variable-size window

**Related:** [[Array]] [[String]]

---

## 💡 Python Code Snippet

```python
def sliding_window_fixed(arr, k):
    """
    Fixed-size sliding window example
    Find maximum sum of k consecutive elements
    """
    if len(arr) < k:
        return None
    
    # Calculate sum of first window
    window_sum = sum(arr[:k])
    max_sum = window_sum
    
    # Slide the window: remove leftmost, add rightmost
    for i in range(k, len(arr)):
        window_sum = window_sum - arr[i - k] + arr[i]
        max_sum = max(max_sum, window_sum)
    
    return max_sum

def sliding_window_variable(s, k):
    """
    Variable-size sliding window example
    Find longest substring with at most k distinct characters
    """
    if k == 0:
        return 0
    
    char_count = {}
    left = 0
    max_length = 0
    
    for right in range(len(s)):
        # Expand window by including character at right
        char_count[s[right]] = char_count.get(s[right], 0) + 1
        
        # Shrink window if we exceed k distinct characters
        while len(char_count) > k:
            char_count[s[left]] -= 1
            if char_count[s[left]] == 0:
                del char_count[s[left]]
            left += 1
        
        # Update maximum length
        max_length = max(max_length, right - left + 1)
    
    return max_length
```

---

## 🔗 LeetCode Practice Problems (10)

- **Maximum Average Subarray I** – https://leetcode.com/problems/maximum-average-subarray-i/
- **Minimum Window Substring** – https://leetcode.com/problems/minimum-window-substring/
- **Longest Substring Without Repeating Characters** – https://leetcode.com/problems/longest-substring-without-repeating-characters/
- **Sliding Window Maximum** – https://leetcode.com/problems/sliding-window-maximum/
- **Longest Substring with At Most K Distinct Characters** – https://leetcode.com/problems/longest-substring-with-at-most-k-distinct-characters/
- **Fruit Into Baskets** – https://leetcode.com/problems/fruit-into-baskets/
- **Longest Repeating Character Replacement** – https://leetcode.com/problems/longest-repeating-character-replacement/
- **Permutation in String** – https://leetcode.com/problems/permutation-in-string/
- **Find All Anagrams in a String** – https://leetcode.com/problems/find-all-anagrams-in-a-string/
- **Subarray Product Less Than K** – https://leetcode.com/problems/subarray-product-less-than-k/

---

## 🧠 Flashcard (for spaced repetition)

What is the Sliding Window pattern? ::

• A technique that maintains a window (subarray/substring) that slides through data to avoid nested loops
• Reduces time complexity from O(n²) to O(n) for contiguous subarray/substring problems
• Two types: fixed-size (constant window) and variable-size (expand/shrink based on conditions)

[[sliding-window]] 