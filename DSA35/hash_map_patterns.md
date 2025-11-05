# Hash Map/Set Patterns

## Explanation

Hash Map/Set patterns utilize hash tables for O(1) average time complexity operations like insert, delete, and lookup. These patterns are essential for problems involving counting, deduplication, two-sum variations, and efficient lookups. The key is recognizing when constant-time access can optimize the solution.

## Algorithm (Word-based)

- Identify the need for efficient lookups, counting, or deduplication
- Choose appropriate hash structure (set for uniqueness, map for key-value)
- Process elements and store relevant information in hash structure
- Use hash structure to quickly find required information
- Maintain hash structure as needed throughout the process

## Code Template

```python
def two_sum(nums, target):
    """
    Two Sum using hash map for O(1) lookups
    """
    num_map = {}  # Value to index mapping
    
    for i, num in enumerate(nums):
        complement = target - num
        if complement in num_map:
            return [num_map[complement], i]
        num_map[num] = i
    
    return []  # No solution found

def contains_duplicate(nums):
    """
    Check for duplicates using hash set
    """
    seen = set()
    for num in nums:
        if num in seen:
            return True
        seen.add(num)
    return False

def group_anagrams(strs):
    """
    Group anagrams using hash map with sorted string as key
    """
    anagram_groups = {}
    
    for s in strs:
        # Sort characters to create canonical form
        key = ''.join(sorted(s))
        
        if key not in anagram_groups:
            anagram_groups[key] = []
        anagram_groups[key].append(s)
    
    return list(anagram_groups.values())

def longest_consecutive_sequence(nums):
    """
    Find longest consecutive sequence using hash set
    """
    if not nums:
        return 0
    
    num_set = set(nums)
    longest = 0
    
    for num in num_set:
        # Only start counting if num-1 doesn't exist
        if num - 1 not in num_set:
            current_num = num
            current_length = 1
            
            # Count consecutive numbers
            while current_num + 1 in num_set:
                current_num += 1
                current_length += 1
            
            longest = max(longest, current_length)
    
    return longest

def first_unique_char(s):
    """
    Find first non-repeating character using hash map for counting
    """
    char_count = {}
    
    # Count frequency of each character
    for char in s:
        char_count[char] = char_count.get(char, 0) + 1
    
    # Find first character with count of 1
    for i, char in enumerate(s):
        if char_count[char] == 1:
            return i
    
    return -1
```

## Practice Questions

1. [Two Sum](https://leetcode.com/problems/two-sum/) - Easy
2. [Group Anagrams](https://leetcode.com/problems/group-anagrams/) - Medium
3. [Valid Anagram](https://leetcode.com/problems/valid-anagram/) - Easy
4. [Contains Duplicate](https://leetcode.com/problems/contains-duplicate/) - Easy
5. [Longest Consecutive Sequence](https://leetcode.com/problems/longest-consecutive-sequence/) - Medium
6. [First Unique Character in a String](https://leetcode.com/problems/first-unique-character-in-a-string/) - Easy
7. [Find All Anagrams in a String](https://leetcode.com/problems/find-all-anagrams-in-a-string/) - Medium
8. [Group Shifted Strings](https://leetcode.com/problems/group-shifted-strings/) - Medium
9. [Word Pattern](https://leetcode.com/problems/word-pattern/) - Easy
10. [Two Sum III - Data structure design](https://leetcode.com/problems/two-sum-iii-data-structure-design/) - Easy

## Notes

- Hash maps provide O(1) average time complexity for operations
- Use hash sets for uniqueness checking
- Can solve many problems that would otherwise require sorting

---

**Tags:** #dsa #hash-map #hash-set #lookup #leetcode

**Last Reviewed:** 2025-11-05