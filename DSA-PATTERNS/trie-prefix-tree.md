## 🧠 Trie (Prefix Tree)

**Trie (Prefix Tree)** is a tree-like data structure that stores strings efficiently by sharing common prefixes among multiple strings.

• Each node represents a character, and paths from root to leaves represent complete words
• Efficient for prefix-based operations: insert, search, and startsWith all O(m) where m is string length
• Space-efficient for storing many strings with common prefixes
• Used for autocomplete, spell checkers, and word games

**Related:** [[Tree]] [[String]] [[Prefix]] [[Search]]

---

## 💡 Python Code Snippet

```python
class TrieNode:
    def __init__(self):
        self.children = {}  # Dictionary to store child nodes
        self.is_end_of_word = False  # Mark end of a valid word

class Trie:
    """
    Basic Trie implementation for word storage and retrieval
    """
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        """Insert a word into the trie"""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
    
    def search(self, word):
        """Search if word exists in trie"""
        node = self._find_node(word)
        return node is not None and node.is_end_of_word
    
    def starts_with(self, prefix):
        """Check if any word starts with given prefix"""
        return self._find_node(prefix) is not None
    
    def _find_node(self, prefix):
        """Helper method to find node for given prefix"""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node

def word_search_ii_optimized(board, words):
    """
    Find all words in board using Trie for optimization
    """
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.word = None
    
    def build_trie():
        root = TrieNode()
        for word in words:
            node = root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.word = word
        return root
    
    def backtrack(row, col, parent):
        char = board[row][col]
        current_node = parent.children[char]
        
        # Check if we found a word
        if current_node.word:
            result.add(current_node.word)
        
        # Mark cell as visited
        board[row][col] = '#'
        
        # Explore all 4 directions
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            
            if (0 <= new_row < len(board) and 0 <= new_col < len(board[0]) and
                board[new_row][new_col] in current_node.children):
                
                backtrack(new_row, new_col, current_node)
        
        # Restore cell
        board[row][col] = char
        
        # Optimization: remove leaf nodes to prune search space
        if not current_node.children:
            parent.children.pop(char)
    
    root = build_trie()
    result = set()
    
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] in root.children:
                backtrack(i, j, root)
    
    return list(result)

class AutocompleteSystem:
    """
    Design search autocomplete system using Trie
    """
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.sentences = []  # Store sentences that pass through this node
    
    def __init__(self, sentences, times):
        self.root = self.TrieNode()
        self.current_node = self.root
        self.current_sentence = ""
        
        # Build trie with initial data
        for i, sentence in enumerate(sentences):
            self._insert(sentence, times[i])
    
    def _insert(self, sentence, frequency):
        node = self.root
        for char in sentence:
            if char not in node.children:
                node.children[char] = self.TrieNode()
            node = node.children[char]
            node.sentences.append((frequency, sentence))
            node.sentences.sort(key=lambda x: (-x[0], x[1]))  # Sort by frequency desc, sentence asc
            if len(node.sentences) > 3:
                node.sentences.pop()
    
    def input(self, c):
        if c == '#':
            # End of sentence, add to trie
            self._insert(self.current_sentence, 1)
            self.current_sentence = ""
            self.current_node = self.root
            return []
        
        self.current_sentence += c
        if c in self.current_node.children:
            self.current_node = self.current_node.children[c]
            return [sentence for _, sentence in self.current_node.sentences]
        else:
            self.current_node = self.TrieNode()  # Dead end
            return []

def replace_words(dictionary, sentence):
    """
    Replace words with their roots using Trie
    """
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.is_word = False
    
    # Build trie with dictionary roots
    root = TrieNode()
    for word in dictionary:
        node = root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_word = True
    
    def find_root(word):
        node = root
        for i, char in enumerate(word):
            if char not in node.children:
                return word
            node = node.children[char]
            if node.is_word:
                return word[:i + 1]
        return word
    
    words = sentence.split()
    return ' '.join(find_root(word) for word in words)

def maximum_xor_trie(nums):
    """
    Find maximum XOR of two numbers using Trie
    """
    class TrieNode:
        def __init__(self):
            self.children = {}
    
    def insert_number(root, num):
        node = root
        for i in range(31, -1, -1):
            bit = (num >> i) & 1
            if bit not in node.children:
                node.children[bit] = TrieNode()
            node = node.children[bit]
    
    def find_max_xor(root, num):
        node = root
        max_xor = 0
        
        for i in range(31, -1, -1):
            bit = (num >> i) & 1
            # Try to go opposite direction for maximum XOR
            opposite_bit = 1 - bit
            
            if opposite_bit in node.children:
                max_xor |= (1 << i)
                node = node.children[opposite_bit]
            else:
                node = node.children[bit]
        
        return max_xor
    
    root = TrieNode()
    
    # Insert all numbers into trie
    for num in nums:
        insert_number(root, num)
    
    max_result = 0
    for num in nums:
        max_result = max(max_result, find_max_xor(root, num))
    
    return max_result

class MapSum:
    """
    Map sum pairs using Trie
    """
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.value = 0
    
    def __init__(self):
        self.root = self.TrieNode()
        self.key_values = {}  # Track key-value pairs
    
    def insert(self, key, val):
        # Calculate delta if key already exists
        delta = val - self.key_values.get(key, 0)
        self.key_values[key] = val
        
        # Update trie with delta
        node = self.root
        for char in key:
            if char not in node.children:
                node.children[char] = self.TrieNode()
            node = node.children[char]
            node.value += delta
    
    def sum(self, prefix):
        # Find node for prefix
        node = self.root
        for char in prefix:
            if char not in node.children:
                return 0
            node = node.children[char]
        return node.value

def word_squares(words):
    """
    Find all word squares using Trie
    """
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.word_indices = []  # Indices of words passing through this node
    
    def build_trie():
        root = TrieNode()
        for i, word in enumerate(words):
            node = root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
                node.word_indices.append(i)
        return root
    
    def get_words_with_prefix(prefix):
        node = root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        return [words[i] for i in node.word_indices]
    
    def backtrack(square):
        if len(square) == len(words[0]):
            result.append(square[:])
            return
        
        # Find prefix for next row based on current column
        row = len(square)
        prefix = ''.join(square[i][row] for i in range(row))
        
        for word in get_words_with_prefix(prefix):
            square.append(word)
            backtrack(square)
            square.pop()
    
    if not words:
        return []
    
    root = build_trie()
    result = []
    
    # Try each word as first row
    for word in words:
        backtrack([word])
    
    return result

def palindrome_pairs_trie(words):
    """
    Find palindrome pairs using Trie
    """
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.word_index = -1
            self.palindrome_suffixes = []  # Indices where remaining suffix is palindrome
    
    def is_palindrome(s):
        return s == s[::-1]
    
    def build_trie():
        root = TrieNode()
        for i, word in enumerate(words):
            node = root
            # Insert word in reverse order
            for j, char in enumerate(reversed(word)):
                # Check if remaining prefix is palindrome
                remaining = word[:len(word) - j - 1]
                if is_palindrome(remaining):
                    node.palindrome_suffixes.append(i)
                
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            
            node.word_index = i
            node.palindrome_suffixes.append(i)  # Empty string is palindrome
        
        return root
    
    root = build_trie()
    result = []
    
    for i, word in enumerate(words):
        node = root
        
        # Case 1: word + word[j] where word[j] is complete word
        for j, char in enumerate(word):
            if node.word_index != -1 and node.word_index != i:
                remaining = word[j:]
                if is_palindrome(remaining):
                    result.append([i, node.word_index])
            
            if char not in node.children:
                break
            node = node.children[char]
        else:
            # Case 2: word + suffix where suffix makes palindrome
            if node.word_index != -1 and node.word_index != i:
                result.append([i, node.word_index])
            
            # Case 3: word + word[j] where remaining forms palindrome
            for j in node.palindrome_suffixes:
                if j != i:
                    result.append([i, j])
    
    return result
```

---

## 🔗 LeetCode Practice Problems (10)

- **Implement Trie (Prefix Tree)** – https://leetcode.com/problems/implement-trie-prefix-tree/
- **Word Search II** – https://leetcode.com/problems/word-search-ii/
- **Design Add and Search Words Data Structure** – https://leetcode.com/problems/design-add-and-search-words-data-structure/
- **Replace Words** – https://leetcode.com/problems/replace-words/
- **Maximum XOR of Two Numbers in an Array** – https://leetcode.com/problems/maximum-xor-of-two-numbers-in-an-array/
- **Map Sum Pairs** – https://leetcode.com/problems/map-sum-pairs/
- **Word Squares** – https://leetcode.com/problems/word-squares/
- **Palindrome Pairs** – https://leetcode.com/problems/palindrome-pairs/
- **Design Search Autocomplete System** – https://leetcode.com/problems/design-search-autocomplete-system/
- **Stream of Characters** – https://leetcode.com/problems/stream-of-characters/

---

## 🧠 Flashcard (for spaced repetition)

What is the Trie (Prefix Tree) pattern? :: • Tree structure storing strings efficiently by sharing common prefixes with each node representing a character • Provides O(m) operations for insert, search, and prefix queries where m is string length • Used for autocomplete, spell checkers, word games, and prefix-based string operations [[trie-prefix-tree]] 