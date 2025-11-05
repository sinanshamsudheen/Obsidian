# Trie

## Explanation

A Trie (prefix tree) is a tree-like data structure that stores a dynamic set of strings where keys are usually strings. Each node represents a character of the string, and paths from root to nodes represent prefixes. Tries are efficient for prefix-based operations, autocomplete, spell check, and dictionary implementations.

## Algorithm (Word-based)

- Create root node with empty value
- For each word insertion, traverse character by character
- If character doesn't exist as child, create new node
- Mark end of word with special flag
- For search, follow path and check if end flag exists

## Code Template

```python
class TrieNode:
    def __init__(self):
        self.children = {}  # Map from character to TrieNode
        self.is_end = False  # Marks end of a word

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        """
        Insert a word into the trie
        """
        current = self.root
        
        for char in word:
            if char not in current.children:
                current.children[char] = TrieNode()
            current = current.children[char]
        
        current.is_end = True  # Mark end of word
    
    def search(self, word):
        """
        Search for a complete word in the trie
        """
        current = self.root
        
        for char in word:
            if char not in current.children:
                return False
            current = current.children[char]
        
        return current.is_end
    
    def starts_with(self, prefix):
        """
        Check if any word in trie starts with the given prefix
        """
        current = self.root
        
        for char in prefix:
            if char not in current.children:
                return False
            current = current.children[char]
        
        return True
    
    def get_words_with_prefix(self, prefix):
        """
        Get all words that start with the given prefix
        """
        current = self.root
        
        # Navigate to the prefix
        for char in prefix:
            if char not in current.children:
                return []
            current = current.children[char]
        
        # Collect all words from this point
        words = []
        self._dfs_collect_words(current, prefix, words)
        return words
    
    def _dfs_collect_words(self, node, current_word, words):
        """
        Helper method to collect all words from a given node
        """
        if node.is_end:
            words.append(current_word)
        
        for char, child_node in node.children.items():
            self._dfs_collect_words(child_node, current_word + char, words)

# Advanced trie for specific problems
class WordDictionary:
    """
    Trie with support for wildcard searches
    """
    def __init__(self):
        self.root = TrieNode()
    
    def add_word(self, word):
        current = self.root
        for char in word:
            if char not in current.children:
                current.children[char] = TrieNode()
            current = current.children[char]
        current.is_end = True
    
    def search(self, word):
        def dfs(node, index):
            if index == len(word):
                return node.is_end
            
            char = word[index]
            if char == '.':
                # Wildcard: try all possible children
                for child in node.children.values():
                    if dfs(child, index + 1):
                        return True
                return False
            else:
                # Specific character
                if char not in node.children:
                    return False
                return dfs(node.children[char], index + 1)
        
        return dfs(self.root, 0)
```

## Practice Questions

1. [Implement Trie (Prefix Tree)](https://leetcode.com/problems/implement-trie-prefix-tree/) - Medium
2. [Design Add and Search Words Data Structure](https://leetcode.com/problems/design-add-and-search-words-data-structure/) - Medium
3. [Word Search II](https://leetcode.com/problems/word-search-ii/) - Hard
4. [Longest Word in Dictionary](https://leetcode.com/problems/longest-word-in-dictionary/) - Medium
5. [Replace Words](https://leetcode.com/problems/replace-words/) - Medium
6. [Maximum XOR of Two Numbers in an Array](https://leetcode.com/problems/maximum-xor-of-two-numbers-in-an-array/) - Medium
7. [Palindrome Pairs](https://leetcode.com/problems/palindrome-pairs/) - Hard
8. [Stream of Characters](https://leetcode.com/problems/stream-of-characters/) - Hard
9. [Concatenated Words](https://leetcode.com/problems/concatenated-words/) - Hard
10. [Word Squares](https://leetcode.com/problems/word-squares/) - Hard

## Notes

- Time complexity: O(m) for insert/search where m is word length
- Space complexity: O(ALPHABET_SIZE * N * M) in worst case
- Useful for prefix-based operations and autocomplete systems

---

**Tags:** #dsa #trie #prefix-tree #leetcode

**Last Reviewed:** 2025-11-05