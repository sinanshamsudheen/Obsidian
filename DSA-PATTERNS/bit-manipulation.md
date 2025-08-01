## 🧠 Bit Manipulation

**Bit Manipulation** uses bitwise operations to solve problems efficiently by working directly with binary representations.

• Common operations: AND (&), OR (|), XOR (^), NOT (~), left shift (<<), right shift (>>)
• XOR properties: a^a = 0, a^0 = a, XOR is commutative and associative
• Useful for finding unique numbers, checking power of 2, and optimization problems
• Often provides O(1) space solutions and can be faster than arithmetic operations

**Related:** [[Binary]] [[Mathematics]] [[Optimization]]

---

## 💡 Python Code Snippet

```python
def single_number(nums):
    """
    Find single number in array where every other number appears twice
    Uses XOR property: a^a = 0, a^0 = a
    """
    result = 0
    for num in nums:
        result ^= num
    return result

def single_number_ii(nums):
    """
    Find single number where every other number appears three times
    """
    ones = twos = 0
    
    for num in nums:
        # Update twos first, then ones
        twos |= ones & num
        ones ^= num
        
        # Remove numbers that appear three times
        threes = ones & twos
        ones &= ~threes
        twos &= ~threes
    
    return ones

def two_single_numbers(nums):
    """
    Find two single numbers in array where every other number appears twice
    """
    # XOR all numbers to get XOR of the two unique numbers
    xor_all = 0
    for num in nums:
        xor_all ^= num
    
    # Find rightmost set bit in XOR result
    rightmost_set_bit = xor_all & (-xor_all)
    
    # Divide numbers into two groups and XOR each group
    num1 = num2 = 0
    for num in nums:
        if num & rightmost_set_bit:
            num1 ^= num
        else:
            num2 ^= num
    
    return [num1, num2]

def is_power_of_two(n):
    """
    Check if number is power of 2
    Power of 2 has only one bit set
    """
    return n > 0 and (n & (n - 1)) == 0

def count_set_bits(n):
    """
    Count number of 1s in binary representation
    """
    count = 0
    while n:
        count += 1
        n &= n - 1  # Remove rightmost set bit
    return count

def reverse_bits(n):
    """
    Reverse bits of a 32-bit unsigned integer
    """
    result = 0
    for _ in range(32):
        result = (result << 1) | (n & 1)
        n >>= 1
    return result

def find_complement(num):
    """
    Find complement of number (flip all bits up to most significant bit)
    """
    # Find number of bits in num
    bit_length = num.bit_length()
    
    # Create mask with all 1s of same length
    mask = (1 << bit_length) - 1
    
    # XOR with mask to flip bits
    return num ^ mask

def subsets_bit_manipulation(nums):
    """
    Generate all subsets using bit manipulation
    """
    n = len(nums)
    result = []
    
    # Generate all possible combinations (2^n)
    for i in range(1 << n):
        subset = []
        for j in range(n):
            # Check if j-th bit is set
            if i & (1 << j):
                subset.append(nums[j])
        result.append(subset)
    
    return result

def missing_number_xor(nums):
    """
    Find missing number in array [0, n] using XOR
    """
    n = len(nums)
    result = n  # Start with n
    
    for i in range(n):
        result ^= i ^ nums[i]
    
    return result

def get_sum_without_arithmetic(a, b):
    """
    Add two numbers without using + or - operators
    """
    # 32-bit mask for Python (handles negative numbers)
    mask = 0xFFFFFFFF
    
    while b:
        # Calculate carry
        carry = (a & b) << 1
        
        # Calculate sum without carry
        a = (a ^ b) & mask
        
        # Update b to carry
        b = carry & mask
    
    # Handle negative numbers in Python
    return a if a <= 0x7FFFFFFF else ~(a ^ mask)

def range_bitwise_and(left, right):
    """
    Find bitwise AND of all numbers in range [left, right]
    """
    shift = 0
    
    # Find common prefix of left and right
    while left != right:
        left >>= 1
        right >>= 1
        shift += 1
    
    # Shift back to get result
    return left << shift

def maximum_xor(nums):
    """
    Find maximum XOR of any two numbers in array
    Using Trie approach for efficiency
    """
    class TrieNode:
        def __init__(self):
            self.children = {}
    
    def insert(root, num):
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
            # Try to go to opposite bit for maximum XOR
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
        insert(root, num)
    
    max_result = 0
    for num in nums:
        max_result = max(max_result, find_max_xor(root, num))
    
    return max_result
```

---

## 🔗 LeetCode Practice Problems (10)

- **Single Number** – https://leetcode.com/problems/single-number/
- **Single Number II** – https://leetcode.com/problems/single-number-ii/
- **Single Number III** – https://leetcode.com/problems/single-number-iii/
- **Number of 1 Bits** – https://leetcode.com/problems/number-of-1-bits/
- **Reverse Bits** – https://leetcode.com/problems/reverse-bits/
- **Power of Two** – https://leetcode.com/problems/power-of-two/
- **Missing Number** – https://leetcode.com/problems/missing-number/
- **Sum of Two Integers** – https://leetcode.com/problems/sum-of-two-integers/
- **Maximum XOR of Two Numbers in an Array** – https://leetcode.com/problems/maximum-xor-of-two-numbers-in-an-array/
- **Subsets** – https://leetcode.com/problems/subsets/

---

## 🧠 Flashcard (for spaced repetition)

What is the Bit Manipulation pattern? ::

• Uses bitwise operations (AND, OR, XOR, shifts) to solve problems efficiently with binary representations  
• Key XOR properties: a^a=0, a^0=a; useful for finding unique numbers and duplicates
• Often provides O(1) space solutions and can be faster than arithmetic operations

[[bit-manipulation]] 