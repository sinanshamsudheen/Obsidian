# Depth First Search (Graph)

## Explanation

Graph DFS explores as far as possible along each branch before backtracking, using a stack (explicitly or recursively). It's used for exploring all reachable nodes from a source, detecting cycles, finding connected components, and path finding. This pattern is fundamental for solving graph problems that require complete exploration.

## Algorithm (Word-based)

- Initialize visited set to track explored nodes
- Start from source node
- Mark current node as visited
- Explore all unvisited neighbors recursively
- For each neighbor, repeat the process
- Backtrack when no more unvisited neighbors exist

## Code Template

```python
def dfs_graph(graph, start):
    """
    Depth-first search in a graph
    """
    visited = set()
    result = []
    
    def dfs(node):
        if node in visited:
            return
        
        visited.add(node)
        result.append(node)  # Process current node
        
        # Explore all neighbors
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor)
    
    dfs(start)
    return result

# Iterative DFS using stack
def dfs_graph_iterative(graph, start):
    """
    Iterative depth-first search in a graph
    """
    visited = set()
    stack = [start]
    result = []
    
    while stack:
        node = stack.pop()
        
        if node not in visited:
            visited.add(node)
            result.append(node)
            
            # Add neighbors to stack (in reverse order for consistent traversal)
            for neighbor in reversed(graph[node]):
                if neighbor not in visited:
                    stack.append(neighbor)
    
    return result
```

## Practice Questions

1. [Number of Islands](https://leetcode.com/problems/number-of-islands/) - Medium
2. [Max Area of Island](https://leetcode.com/problems/max-area-of-island/) - Medium
3. [Clone Graph](https://leetcode.com/problems/clone-graph/) - Medium
4. [Number of Connected Components in an Undirected Graph](https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/) - Medium
5. [Course Schedule](https://leetcode.com/problems/course-schedule/) - Medium
6. [Course Schedule II](https://leetcode.com/problems/course-schedule-ii/) - Medium
7. [Pacific Atlantic Water Flow](https://leetcode.com/problems/pacific-atlantic-water-flow/) - Medium
8. [Number of Provinces](https://leetcode.com/problems/number-of-provinces/) - Medium
9. [Reconstruct Itinerary](https://leetcode.com/problems/reconstruct-itinerary/) - Hard
10. [All Paths From Source to Target](https://leetcode.com/problems/all-paths-from-source-to-target/) - Medium

## Notes

- Can be implemented recursively or iteratively
- Useful for finding connected components
- Good for exploring all paths in a graph

---

**Tags:** #dsa #graph-dfs #leetcode

**Last Reviewed:** 2025-11-05