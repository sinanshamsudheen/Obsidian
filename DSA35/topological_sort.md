# Topological Sort

## Explanation

Topological sort arranges vertices in a directed acyclic graph (DAG) such that for every directed edge from vertex A to B, A comes before B in the ordering. This pattern is essential for dependency resolution problems like course scheduling, build systems, and task ordering.

## Algorithm (Word-based)

- Calculate in-degree (number of incoming edges) for each vertex
- Add all vertices with in-degree 0 to queue
- While queue is not empty
- Dequeue a vertex and add to result
- Decrease in-degree of all adjacent vertices
- Add any vertex with in-degree 0 to queue
- Return result if all vertices processed, else return empty (cycle exists)

## Code Template

```python
from collections import deque, defaultdict

def topological_sort(vertices, edges):
    """
    Topological sort using Kahn's algorithm
    """
    # Build graph and calculate in-degrees
    graph = defaultdict(list)
    in_degree = {i: 0 for i in range(vertices)}
    
    for parent, child in edges:
        graph[parent].append(child)
        in_degree[child] += 1
    
    # Find all sources (vertices with in-degree 0)
    sources = deque()
    for vertex in in_degree:
        if in_degree[vertex] == 0:
            sources.append(vertex)
    
    sorted_order = []
    
    while sources:
        vertex = sources.popleft()
        sorted_order.append(vertex)
        
        # Decrement in-degree for each child
        for child in graph[vertex]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                sources.append(child)
    
    # Check if there's a cycle
    if len(sorted_order) != vertices:
        return []  # Cycle exists, no valid topological sort
    
    return sorted_order
```

## Practice Questions

1. [Course Schedule](https://leetcode.com/problems/course-schedule/) - Medium
2. [Course Schedule II](https://leetcode.com/problems/course-schedule-ii/) - Medium
3. [Alien Dictionary](https://leetcode.com/problems/alien-dictionary/) - Hard
4. [Minimum Height Trees](https://leetcode.com/problems/minimum-height-trees/) - Medium
5. [Sequence Reconstruction](https://leetcode.com/problems/sequence-reconstruction/) - Medium
6. [Parallel Courses](https://leetcode.com/problems/parallel-courses/) - Medium
7. [Build a Matrix With Conditions](https://leetcode.com/problems/build-a-matrix-with-conditions/) - Hard
8. [Find Eventual Safe States](https://leetcode.com/problems/find-eventual-safe-states/) - Medium
9. [Course Schedule III](https://leetcode.com/problems/course-schedule-iii/) - Hard
10. [Number of Restricted Paths From First to Last Node](https://leetcode.com/problems/number-of-restricted-paths-from-first-to-last-node/) - Hard

## Notes

- Only works for directed acyclic graphs (DAGs)
- Kahn's algorithm uses in-degrees
- Can also be implemented with DFS and post-order traversal

---

**Tags:** #dsa #topological-sort #leetcode

**Last Reviewed:** 2025-11-05