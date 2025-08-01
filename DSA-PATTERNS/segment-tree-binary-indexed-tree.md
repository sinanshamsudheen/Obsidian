## 🧠 Segment Tree / Binary Indexed Tree (Fenwick Tree)

**Segment Tree / Binary Indexed Tree** are tree-based data structures that support efficient range queries and updates on arrays.

• Segment Tree: Supports range queries (sum, min, max) and updates in O(log n) time
• Binary Indexed Tree (Fenwick Tree): Specialized for prefix sums with simpler implementation
• Both use tree structures to divide array into segments for efficient processing
• Essential for problems requiring frequent range operations on dynamic arrays

**Related:** [[Tree]] [[Array]] [[Range Queries]] [[Data Structure]]

---

## 💡 Python Code Snippet

```python
class SegmentTree:
    """
    Generic Segment Tree supporting range queries and point updates
    """
    def __init__(self, arr, operation='sum'):
        self.n = len(arr)
        self.tree = [0] * (4 * self.n)
        self.operation = operation
        self.arr = arr[:]
        
        if operation == 'sum':
            self.combine = lambda a, b: a + b
            self.identity = 0
        elif operation == 'min':
            self.combine = lambda a, b: min(a, b)
            self.identity = float('inf')
        elif operation == 'max':
            self.combine = lambda a, b: max(a, b)
            self.identity = float('-inf')
        
        self.build(1, 0, self.n - 1)
    
    def build(self, node, start, end):
        """Build the segment tree"""
        if start == end:
            self.tree[node] = self.arr[start]
        else:
            mid = (start + end) // 2
            self.build(2 * node, start, mid)
            self.build(2 * node + 1, mid + 1, end)
            self.tree[node] = self.combine(self.tree[2 * node], 
                                         self.tree[2 * node + 1])
    
    def update(self, idx, value):
        """Update value at index idx"""
        self._update(1, 0, self.n - 1, idx, value)
    
    def _update(self, node, start, end, idx, value):
        if start == end:
            self.tree[node] = value
            self.arr[idx] = value
        else:
            mid = (start + end) // 2
            if idx <= mid:
                self._update(2 * node, start, mid, idx, value)
            else:
                self._update(2 * node + 1, mid + 1, end, idx, value)
            
            self.tree[node] = self.combine(self.tree[2 * node], 
                                         self.tree[2 * node + 1])
    
    def query(self, left, right):
        """Query range [left, right]"""
        return self._query(1, 0, self.n - 1, left, right)
    
    def _query(self, node, start, end, left, right):
        if right < start or end < left:
            return self.identity
        
        if left <= start and end <= right:
            return self.tree[node]
        
        mid = (start + end) // 2
        left_result = self._query(2 * node, start, mid, left, right)
        right_result = self._query(2 * node + 1, mid + 1, end, left, right)
        
        return self.combine(left_result, right_result)

class LazySegmentTree:
    """
    Segment Tree with lazy propagation for range updates
    """
    def __init__(self, arr):
        self.n = len(arr)
        self.tree = [0] * (4 * self.n)
        self.lazy = [0] * (4 * self.n)
        self.arr = arr[:]
        self.build(1, 0, self.n - 1)
    
    def build(self, node, start, end):
        if start == end:
            self.tree[node] = self.arr[start]
        else:
            mid = (start + end) // 2
            self.build(2 * node, start, mid)
            self.build(2 * node + 1, mid + 1, end)
            self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]
    
    def push(self, node, start, end):
        """Push lazy value down to children"""
        if self.lazy[node] != 0:
            self.tree[node] += (end - start + 1) * self.lazy[node]
            
            if start != end:  # Not leaf node
                self.lazy[2 * node] += self.lazy[node]
                self.lazy[2 * node + 1] += self.lazy[node]
            
            self.lazy[node] = 0
    
    def range_update(self, left, right, value):
        """Add value to range [left, right]"""
        self._range_update(1, 0, self.n - 1, left, right, value)
    
    def _range_update(self, node, start, end, left, right, value):
        self.push(node, start, end)
        
        if start > right or end < left:
            return
        
        if start >= left and end <= right:
            self.lazy[node] += value
            self.push(node, start, end)
            return
        
        mid = (start + end) // 2
        self._range_update(2 * node, start, mid, left, right, value)
        self._range_update(2 * node + 1, mid + 1, end, left, right, value)
        
        self.push(2 * node, start, mid)
        self.push(2 * node + 1, mid + 1, end)
        
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]
    
    def range_query(self, left, right):
        """Query sum of range [left, right]"""
        return self._range_query(1, 0, self.n - 1, left, right)
    
    def _range_query(self, node, start, end, left, right):
        if start > right or end < left:
            return 0
        
        self.push(node, start, end)
        
        if start >= left and end <= right:
            return self.tree[node]
        
        mid = (start + end) // 2
        left_sum = self._range_query(2 * node, start, mid, left, right)
        right_sum = self._range_query(2 * node + 1, mid + 1, end, left, right)
        
        return left_sum + right_sum

class BinaryIndexedTree:
    """
    Binary Indexed Tree (Fenwick Tree) for prefix sum queries
    """
    def __init__(self, n):
        self.n = n
        self.tree = [0] * (n + 1)
    
    def update(self, idx, delta):
        """Add delta to element at index idx (1-indexed)"""
        while idx <= self.n:
            self.tree[idx] += delta
            idx += idx & (-idx)  # Add least significant bit
    
    def prefix_sum(self, idx):
        """Get prefix sum up to index idx (1-indexed)"""
        result = 0
        while idx > 0:
            result += self.tree[idx]
            idx -= idx & (-idx)  # Remove least significant bit
        return result
    
    def range_sum(self, left, right):
        """Get sum of range [left, right] (1-indexed)"""
        return self.prefix_sum(right) - self.prefix_sum(left - 1)
    
    def find_kth_element(self, k):
        """Find index of k-th smallest element (if tree stores frequencies)"""
        idx = 0
        bit_mask = 1
        
        # Find highest power of 2 <= n
        while bit_mask <= self.n:
            bit_mask <<= 1
        bit_mask >>= 1
        
        while bit_mask > 0:
            next_idx = idx + bit_mask
            if next_idx <= self.n and k >= self.tree[next_idx]:
                idx = next_idx
                k -= self.tree[idx]
            bit_mask >>= 1
        
        return idx + 1

class RangeSumQuery:
    """
    Range Sum Query with updates using BIT
    """
    def __init__(self, nums):
        self.nums = nums[:]
        self.bit = BinaryIndexedTree(len(nums))
        
        # Initialize BIT with original values
        for i, num in enumerate(nums):
            self.bit.update(i + 1, num)
    
    def update(self, index, val):
        """Update nums[index] to val"""
        delta = val - self.nums[index]
        self.nums[index] = val
        self.bit.update(index + 1, delta)
    
    def sumRange(self, left, right):
        """Return sum of nums[left:right+1]"""
        return self.bit.range_sum(left + 1, right + 1)

class CountInversions:
    """
    Count inversions in array using BIT
    """
    def __init__(self, nums):
        # Coordinate compression
        sorted_unique = sorted(set(nums))
        self.coord_map = {val: i + 1 for i, val in enumerate(sorted_unique)}
        self.bit = BinaryIndexedTree(len(sorted_unique))
    
    def count_inversions(self, nums):
        """Count number of inversions in array"""
        inversions = 0
        
        for num in nums:
            compressed = self.coord_map[num]
            
            # Count elements greater than current number seen so far
            total_seen = self.bit.prefix_sum(len(self.coord_map))
            smaller_or_equal = self.bit.prefix_sum(compressed)
            inversions += total_seen - smaller_or_equal
            
            # Add current element to BIT
            self.bit.update(compressed, 1)
        
        return inversions

def range_minimum_query_sparse_table(arr):
    """
    Range Minimum Query using Sparse Table (O(1) query, O(n log n) preprocessing)
    """
    n = len(arr)
    k = n.bit_length() - 1
    
    # Build sparse table
    st = [[0] * (k + 1) for _ in range(n)]
    
    # Initialize for length 1
    for i in range(n):
        st[i][0] = arr[i]
    
    # Build for lengths 2, 4, 8, ...
    j = 1
    while (1 << j) <= n:
        i = 0
        while (i + (1 << j) - 1) < n:
            st[i][j] = min(st[i][j-1], st[i + (1 << (j-1))][j-1])
            i += 1
        j += 1
    
    def query(left, right):
        """Query minimum in range [left, right]"""
        length = right - left + 1
        k = length.bit_length() - 1
        return min(st[left][k], st[right - (1 << k) + 1][k])
    
    return query

class SegmentTreeLCA:
    """
    Segment Tree for Lowest Common Ancestor queries
    """
    def __init__(self, tree_edges, root=0):
        self.graph = defaultdict(list)
        for u, v in tree_edges:
            self.graph[u].append(v)
            self.graph[v].append(u)
        
        self.euler_tour = []
        self.first_occurrence = {}
        self.depths = []
        
        self._dfs(root, -1, 0)
        
        # Build RMQ on depths
        self.rmq = SegmentTree(self.depths, 'min')
    
    def _dfs(self, node, parent, depth):
        """DFS to create Euler tour"""
        self.first_occurrence[node] = len(self.euler_tour)
        self.euler_tour.append(node)
        self.depths.append(depth)
        
        for neighbor in self.graph[node]:
            if neighbor != parent:
                self._dfs(neighbor, node, depth + 1)
                self.euler_tour.append(node)
                self.depths.append(depth)
    
    def lca(self, u, v):
        """Find Lowest Common Ancestor of u and v"""
        left = min(self.first_occurrence[u], self.first_occurrence[v])
        right = max(self.first_occurrence[u], self.first_occurrence[v])
        
        min_depth_idx = self.rmq.query(left, right)
        return self.euler_tour[min_depth_idx]

class TwoDimensionalBIT:
    """
    2D Binary Indexed Tree for 2D range sum queries
    """
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.tree = [[0] * (cols + 1) for _ in range(rows + 1)]
    
    def update(self, row, col, delta):
        """Add delta to position (row, col) - 1-indexed"""
        i = row
        while i <= self.rows:
            j = col
            while j <= self.cols:
                self.tree[i][j] += delta
                j += j & (-j)
            i += i & (-i)
    
    def prefix_sum(self, row, col):
        """Get sum of rectangle from (1,1) to (row,col)"""
        result = 0
        i = row
        while i > 0:
            j = col
            while j > 0:
                result += self.tree[i][j]
                j -= j & (-j)
            i -= i & (-i)
        return result
    
    def range_sum(self, row1, col1, row2, col2):
        """Get sum of rectangle from (row1,col1) to (row2,col2) - 1-indexed"""
        return (self.prefix_sum(row2, col2) - 
                self.prefix_sum(row1 - 1, col2) - 
                self.prefix_sum(row2, col1 - 1) + 
                self.prefix_sum(row1 - 1, col1 - 1))

def count_smaller_elements_after_self(nums):
    """
    Count smaller elements after self using BIT
    """
    # Coordinate compression
    sorted_unique = sorted(set(nums))
    coord_map = {val: i + 1 for i, val in enumerate(sorted_unique)}
    
    bit = BinaryIndexedTree(len(sorted_unique))
    result = []
    
    # Process from right to left
    for i in range(len(nums) - 1, -1, -1):
        compressed = coord_map[nums[i]]
        
        # Count elements smaller than current
        smaller_count = bit.prefix_sum(compressed - 1)
        result.append(smaller_count)
        
        # Add current element to BIT
        bit.update(compressed, 1)
    
    return result[::-1]

def reversePairs(nums):
    """
    Count reverse pairs using modified BIT
    """
    # Coordinate compression for 2*nums[j]
    all_vals = nums + [2 * num for num in nums]
    sorted_unique = sorted(set(all_vals))
    coord_map = {val: i + 1 for i, val in enumerate(sorted_unique)}
    
    bit = BinaryIndexedTree(len(sorted_unique))
    count = 0
    
    for i in range(len(nums) - 1, -1, -1):
        # Count how many j > i such that nums[i] > 2 * nums[j]
        target = 2 * nums[i]
        if target in coord_map:
            # Count elements < target
            smaller_count = bit.prefix_sum(coord_map[target] - 1)
            count += smaller_count
        
        # Add current element
        bit.update(coord_map[nums[i]], 1)
    
    return count
```

---

## 🔗 LeetCode Practice Problems (10)

- **Range Sum Query - Mutable** – https://leetcode.com/problems/range-sum-query-mutable/
- **Count of Smaller Numbers After Self** – https://leetcode.com/problems/count-of-smaller-numbers-after-self/
- **Reverse Pairs** – https://leetcode.com/problems/reverse-pairs/
- **Range Sum Query 2D - Mutable** – https://leetcode.com/problems/range-sum-query-2d-mutable/
- **Count of Range Sum** – https://leetcode.com/problems/count-of-range-sum/
- **My Calendar III** – https://leetcode.com/problems/my-calendar-iii/
- **Falling Squares** – https://leetcode.com/problems/falling-squares/
- **The Skyline Problem** – https://leetcode.com/problems/the-skyline-problem/
- **Range Module** – https://leetcode.com/problems/range-module/
- **Maximum Frequency Stack** – https://leetcode.com/problems/maximum-frequency-stack/

---

## 🧠 Flashcard (for spaced repetition)

What is the Segment Tree / Binary Indexed Tree pattern? :: • Tree-based structures for efficient range queries and updates in O(log n) time • Segment Tree: general range operations (sum, min, max), BIT: specialized for prefix sums • Used for dynamic arrays requiring frequent range operations and point/range updates [[segment-tree-binary-indexed-tree]] 