# Union Find (Disjoint Set)

## Explanation

Union-Find (Disjoint Set) is a data structure that keeps track of elements partitioned into non-overlapping subsets. It supports two main operations: 'find' to determine which subset an element belongs to, and 'union' to merge two subsets. It's used for dynamic connectivity problems and cycle detection in graphs.

## Algorithm (Word-based)

- Initialize parent array where each element is its own parent
- For 'find' operation, recursively find root parent with path compression
- For 'union' operation, connect root of one set to root of another
- Use union by rank/size to optimize tree height
- Path compression flattens tree during find operations

## Code Template

```python
class UnionFind:
    def __init__(self, n):
        """
        Initialize Union-Find with n elements
        """
        self.parent = list(range(n))  # Each element is its own parent
        self.rank = [0] * n  # Rank for union by rank optimization
        self.count = n  # Number of connected components
    
    def find(self, x):
        """
        Find root parent of x with path compression
        """
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x, y):
        """
        Union x and y sets
        """
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False  # Already in same set
        
        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
        
        self.count -= 1  # Decrease number of components
        return True
    
    def is_connected(self, x, y):
        """
        Check if x and y are in the same set
        """
        return self.find(x) == self.find(y)

# Usage example
def solve_union_find_problem(edges, n):
    """
    Example problem: detect cycle in undirected graph
    """
    uf = UnionFind(n)
    
    for u, v in edges:
        if not uf.union(u, v):  # If union returns False, cycle detected
            return True  # Cycle exists
    
    return False  # No cycle
```

## Practice Questions

1. [Number of Connected Components in an Undirected Graph](https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/) - Medium
2. [Graph Valid Tree](https://leetcode.com/problems/graph-valid-tree/) - Medium
3. [Number of Islands II](https://leetcode.com/problems/number-of-islands-ii/) - Hard
4. [Most Stones Removed with Same Row or Column](https://leetcode.com/problems/most-stones-removed-with-same-row-or-column/) - Medium
5. [Accounts Merge](https://leetcode.com/problems/accounts-merge/) - Medium
6. [Redundant Connection](https://leetcode.com/problems/redundant-connection/) - Medium
7. [Similar String Groups](https://leetcode.com/problems/similar-string-groups/) - Hard
8. [Smallest String With Swaps](https://leetcode.com/problems/smallest-string-with-swaps/) - Medium
9. [Number of Provinces](https://leetcode.com/problems/number-of-provinces/) - Medium
10. [Satisfiability of Equality Equations](https://leetcode.com/problems/satisfiability-of-equality-equations/) - Medium

## Notes

- Path compression and union by rank optimize operations
- Nearly constant time for operations: O(α(n)) where α is inverse Ackermann function
- Useful for dynamic connectivity and clustering problems

---

**Tags:** #dsa #union-find #leetcode

**Last Reviewed:** 2025-11-05