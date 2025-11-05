# Bitwise Manipulation

## Explanation

Bitwise manipulation involves operating on individual bits of numbers using bitwise operators (AND, OR, XOR, NOT, left shift, right shift). This pattern is used for optimizing operations, working with flags, implementing efficient algorithms, and solving problems with mathematical properties related to binary representations.

## Algorithm (Word-based)

- Identify the bit-based pattern needed for the problem
- Use appropriate bitwise operations (AND, OR, XOR, shifts)
- Manipulate bits to achieve the required result
- Consider bit properties like XOR of same numbers being 0
- Use masks to extract or set specific bits

## Code Template

```python
def get_bit(num, i):
    """
    Get the bit at position i (0-indexed from right)
    """
    return (num >> i) & 1

def set_bit(num, i):
    """
    Set the bit at position i to 1
    """
    return num | (1 << i)

def clear_bit(num, i):
    """
    Clear the bit at position i (set to 0)
    """
    return num & ~(1 << i)

def update_bit(num, i, bit_value):
    """
    Update the bit at position i to the given value
    """
    mask = ~(1 << i)
    return (num & mask) | (bit_value << i)

def single_number(nums):
    """
    Find the single number in array where others appear twice
    Uses XOR property: a ^ a = 0, a ^ 0 = a
    """
    result = 0
    for num in nums:
        result ^= num
    return result

def add_without_plus(a, b):
    """
    Add two numbers without using + operator
    Uses XOR for sum without carry and AND + shift for carry
    """
    while b != 0:
        carry = a & b  # Calculate carry
        a = a ^ b      # Sum without carry
        b = carry << 1 # Shift carry left
    return a

def count_ones(n):
    """
    Count number of 1s in binary representation of n
    """
    count = 0
    while n:
        count += 1
        n = n & (n - 1)  # Remove rightmost set bit
    return count

def is_power_of_two(n):
    """
    Check if n is a power of 2
    """
    return n > 0 and (n & (n - 1)) == 0

def next_power_of_two(n):
    """
    Find the next power of 2 greater than or equal to n
    """
    if n == 0:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1
```

## Practice Questions

1. [Single Number](https://leetcode.com/problems/single-number/) - Easy
2. [Single Number II](https://leetcode.com/problems/single-number-ii/) - Medium
3. [Single Number III](https://leetcode.com/problems/single-number-iii/) - Medium
4. [Counting Bits](https://leetcode.com/problems/counting-bits/) - Easy
5. [Reverse Bits](https://leetcode.com/problems/reverse-bits/) - Easy
6. [Number of 1 Bits](https://leetcode.com/problems/number-of-1-bits/) - Easy
7. [Missing Number](https://leetcode.com/problems/missing-number/) - Easy
8. [Sum of Two Integers](https://leetcode.com/problems/sum-of-two-integers/) - Medium
9. [Maximum XOR of Two Numbers in an Array](https://leetcode.com/problems/maximum-xor-of-two-numbers-in-an-array/) - Medium
10. [Bitwise AND of Numbers Range](https://leetcode.com/problems/bitwise-and-of-numbers-range/) - Medium

## Notes

- XOR of a number with itself is 0: a ^ a = 0
- XOR of a number with 0 is the number: a ^ 0 = a
- AND with bit mask helps extract specific bits
- Right shift (>>) divides by 2, left shift (<<) multiplies by 2

---

**Tags:** #dsa #bitwise-manipulation #bitwise-operations #leetcode

**Last Reviewed:** 2025-11-05