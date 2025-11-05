# Breadth First Search (Graph)

## Explanation

Graph BFS explores nodes level by level, starting from a source node and visiting all neighbors before moving to the next level. It uses a queue to maintain the order of exploration and is ideal for finding shortest paths in unweighted graphs, level-order traversal, and exploring reachable nodes from a source.

## Algorithm (Word-based)

- Initialize queue with source node
- Initialize visited set to track explored nodes
- Add source to visited set
- While queue is not empty
- Dequeue a node and process it
- Enqueue all unvisited neighbors
- Mark neighbors as visited

## Code Template

```python
from collections import deque

def bfs_graph(graph, start):
    """
    Breadth-first search in a graph
    """
    visited = set([start])
    queue = deque([start])
    result = []
    
    while queue:
        node = queue.popleft()
        result.append(node)  # Process current node
        
        # Explore all neighbors
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return result

def bfs_shortest_path(graph, start, end):
    """
    BFS to find shortest path between two nodes
    """
    if start == end:
        return [start]
    
    visited = set([start])
    queue = deque([(start, [start])])  # Store node and path to it
    
    while queue:
        node, path = queue.popleft()
        
        for neighbor in graph[node]:
            if neighbor == end:
                return path + [neighbor]  # Found shortest path
            
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    
    return []  # No path found
```

## Practice Questions

1. [Number of Islands](https://leetcode.com/problems/number-of-islands/) - Medium
2. [Rotting Oranges](https://leetcode.com/problems/rotting-oranges/) - Medium
3. [Walls and Gates](https://leetcode.com/problems/walls-and-gates/) - Medium
4. [Shortest Path in Binary Matrix](https://leetcode.com/problems/shortest-path-in-binary-matrix/) - Medium
5. [Word Ladder](https://leetcode.com/problems/word-ladder/) - Hard
6. [Perfect Squares](https://leetcode.com/problems/perfect-squares/) - Medium
7. [Minimum Knight Moves](https://leetcode.com/problems/minimum-knight-moves/) - Medium
8. [Minimum Genetic Mutation](https://leetcode.com/problems/minimum-genetic-mutation/) - Medium
9. [Open the Lock](https://leetcode.com/problems/open-the-lock/) - Medium
10. [01 Matrix](https://leetcode.com/problems/01-matrix/) - Medium

## Notes

- Guarantees shortest path in unweighted graphs
- Uses queue (FIFO) for processing nodes
- Good for finding levels in graph problems

---

**Tags:** #dsa #graph-bfs #leetcode

**Last Reviewed:** 2025-11-05