## 🧠 Union Find / Disjoint Set

**Union Find (Disjoint Set)** is a data structure that efficiently tracks and merges disjoint sets, useful for connectivity and grouping problems.

• Supports two main operations: Union (merge sets) and Find (find set representative)
• Uses path compression and union by rank/size for near O(1) amortized operations
• Excellent for detecting cycles, connected components, and dynamic connectivity
• Common applications: network connectivity, clustering, and minimum spanning trees

**Related:** [[Graph]] [[Tree]] [[Connectivity]] [[Sets]]

---

## 💡 Python Code Snippet

```python
class UnionFind:
    """
    Basic Union Find with path compression and union by rank
    """
    def __init__(self, n):
        self.parent = list(range(n))  # Initially each element is its own parent
        self.rank = [0] * n  # Rank for union by rank optimization
        self.components = n  # Number of connected components
    
    def find(self, x):
        """Find root of x with path compression"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x, y):
        """Union sets containing x and y"""
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x != root_y:
            # Union by rank: attach smaller rank tree under higher rank tree
            if self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            elif self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1
            
            self.components -= 1
            return True
        return False
    
    def connected(self, x, y):
        """Check if x and y are in the same set"""
        return self.find(x) == self.find(y)
    
    def get_components(self):
        """Get number of connected components"""
        return self.components

class UnionFindBySize:
    """
    Union Find with union by size instead of rank
    """
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n  # Size of each component
        self.components = n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x != root_y:
            # Union by size: attach smaller tree under larger tree
            if self.size[root_x] < self.size[root_y]:
                self.parent[root_x] = root_y
                self.size[root_y] += self.size[root_x]
            else:
                self.parent[root_y] = root_x
                self.size[root_x] += self.size[root_y]
            
            self.components -= 1
            return True
        return False
    
    def get_size(self, x):
        """Get size of component containing x"""
        return self.size[self.find(x)]

def number_of_islands(grid):
    """
    Count number of islands using Union Find
    """
    if not grid or not grid[0]:
        return 0
    
    rows, cols = len(grid), len(grid[0])
    
    # Convert 2D coordinates to 1D index
    def get_index(r, c):
        return r * cols + c
    
    # Count land cells and initialize Union Find
    land_count = sum(row.count('1') for row in grid)
    uf = UnionFind(rows * cols)
    uf.components = land_count  # Start with each land cell as separate component
    
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == '1':
                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    if (0 <= ni < rows and 0 <= nj < cols and 
                        grid[ni][nj] == '1'):
                        uf.union(get_index(i, j), get_index(ni, nj))
    
    return uf.components

def accounts_merge(accounts):
    """
    Merge accounts belonging to same person using Union Find
    """
    from collections import defaultdict
    
    email_to_index = {}
    email_to_name = {}
    index = 0
    
    # Assign unique index to each email
    for account in accounts:
        name = account[0]
        for email in account[1:]:
            if email not in email_to_index:
                email_to_index[email] = index
                email_to_name[email] = name
                index += 1
    
    uf = UnionFind(index)
    
    # Union emails within same account
    for account in accounts:
        first_email = account[1]
        first_index = email_to_index[first_email]
        
        for email in account[2:]:
            uf.union(first_index, email_to_index[email])
    
    # Group emails by their root
    groups = defaultdict(list)
    for email, idx in email_to_index.items():
        root = uf.find(idx)
        groups[root].append(email)
    
    # Build result
    result = []
    for emails in groups.values():
        emails.sort()
        name = email_to_name[emails[0]]
        result.append([name] + emails)
    
    return result

def redundant_connection(edges):
    """
    Find edge that creates cycle in tree
    """
    n = len(edges)
    uf = UnionFind(n + 1)
    
    for u, v in edges:
        if uf.connected(u, v):
            return [u, v]  # This edge creates a cycle
        uf.union(u, v)
    
    return []

def most_stones_removed(stones):
    """
    Maximum stones that can be removed
    """
    n = len(stones)
    uf = UnionFind(n)
    
    # Group stones that can reach each other (same row or column)
    for i in range(n):
        for j in range(i + 1, n):
            if stones[i][0] == stones[j][0] or stones[i][1] == stones[j][1]:
                uf.union(i, j)
    
    # Can remove all stones except one from each connected component
    return n - uf.components

def satisfiability_of_equality_equations(equations):
    """
    Check if equality equations are satisfiable
    """
    uf = UnionFind(26)  # 26 letters
    
    # Process equality constraints first
    for equation in equations:
        if equation[1] == '=':
            x = ord(equation[0]) - ord('a')
            y = ord(equation[3]) - ord('a')
            uf.union(x, y)
    
    # Check inequality constraints
    for equation in equations:
        if equation[1] == '!':
            x = ord(equation[0]) - ord('a')
            y = ord(equation[3]) - ord('a')
            if uf.connected(x, y):
                return False
    
    return True

def regions_cut_by_slashes(grid):
    """
    Count regions after cutting grid by slashes
    """
    n = len(grid)
    # Each cell is divided into 4 triangular parts: 0=top, 1=right, 2=bottom, 3=left
    uf = UnionFind(4 * n * n)
    
    def get_index(r, c, part):
        return 4 * (r * n + c) + part
    
    for r in range(n):
        for c in range(n):
            root = 4 * (r * n + c)
            
            # Connect parts within cell based on character
            if grid[r][c] == '/':
                uf.union(root + 0, root + 3)  # top-left
                uf.union(root + 1, root + 2)  # right-bottom
            elif grid[r][c] == '\\':
                uf.union(root + 0, root + 1)  # top-right
                uf.union(root + 2, root + 3)  # bottom-left
            else:  # ' ' - connect all parts
                uf.union(root + 0, root + 1)
                uf.union(root + 1, root + 2)
                uf.union(root + 2, root + 3)
            
            # Connect with neighboring cells
            # Right neighbor
            if c + 1 < n:
                uf.union(get_index(r, c, 1), get_index(r, c + 1, 3))
            
            # Bottom neighbor
            if r + 1 < n:
                uf.union(get_index(r, c, 2), get_index(r + 1, c, 0))
    
    return uf.components

def largest_component_size_by_common_factor(nums):
    """
    Find largest component where elements share common factors
    """
    def get_prime_factors(n):
        factors = []
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.append(d)
                n //= d
            d += 1
        if n > 1:
            factors.append(n)
        return factors
    
    # Create mapping from number to index
    num_to_index = {num: i for i, num in enumerate(nums)}
    uf = UnionFindBySize(len(nums))
    
    # Map each prime factor to the first number that has it
    factor_to_index = {}
    
    for i, num in enumerate(nums):
        for factor in get_prime_factors(num):
            if factor in factor_to_index:
                uf.union(i, factor_to_index[factor])
            else:
                factor_to_index[factor] = i
    
    # Find largest component size
    max_size = 1
    for i in range(len(nums)):
        max_size = max(max_size, uf.get_size(i))
    
    return max_size

def minimum_spanning_tree_kruskal(n, edges):
    """
    Find Minimum Spanning Tree using Kruskal's algorithm with Union Find
    """
    # Sort edges by weight
    edges.sort(key=lambda x: x[2])
    
    uf = UnionFind(n)
    mst_edges = []
    total_weight = 0
    
    for u, v, weight in edges:
        if uf.union(u, v):  # If union succeeds (no cycle created)
            mst_edges.append([u, v, weight])
            total_weight += weight
            
            if len(mst_edges) == n - 1:  # MST complete
                break
    
    return mst_edges, total_weight
```

---

## 🔗 LeetCode Practice Problems (10)

- **Number of Islands** – https://leetcode.com/problems/number-of-islands/
- **Accounts Merge** – https://leetcode.com/problems/accounts-merge/
- **Redundant Connection** – https://leetcode.com/problems/redundant-connection/
- **Most Stones Removed with Same Row or Column** – https://leetcode.com/problems/most-stones-removed-with-same-row-or-column/
- **Satisfiability of Equality Equations** – https://leetcode.com/problems/satisfiability-of-equality-equations/
- **Regions Cut By Slashes** – https://leetcode.com/problems/regions-cut-by-slashes/
- **Number of Islands II** – https://leetcode.com/problems/number-of-islands-ii/
- **Graph Valid Tree** – https://leetcode.com/problems/graph-valid-tree/
- **Largest Component Size by Common Factor** – https://leetcode.com/problems/largest-component-size-by-common-factor/
- **Connecting Cities With Minimum Cost** – https://leetcode.com/problems/connecting-cities-with-minimum-cost/

---

## 🧠 Flashcard (for spaced repetition)

What is the Union Find / Disjoint Set pattern? ::

• Data structure tracking disjoint sets with Union (merge) and Find (representative) operations
• Uses path compression and union by rank/size for near O(1) amortized performance
• Applied to connectivity problems, cycle detection, clustering, and minimum spanning trees

[[union-find-disjoint-set]] 