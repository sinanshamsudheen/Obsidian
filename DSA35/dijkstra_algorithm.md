# Dijkstra's Algorithm

## Explanation

Dijkstra's algorithm finds the shortest path from a source vertex to all other vertices in a weighted graph with non-negative edge weights. It's a greedy algorithm that uses a priority queue to always process the vertex with the minimum distance from the source first, ensuring optimal results.

## Algorithm (Word-based)

- Initialize distances with infinity, set source distance to 0
- Create priority queue with source vertex and distance 0
- While priority queue is not empty
- Extract vertex with minimum distance
- For each neighbor, update distance if shorter path found
- Add updated neighbors to priority queue
- Return distances array

## Code Template

```python
import heapq
from collections import defaultdict

def dijkstra(graph, start, n):
    """
    Dijkstra's algorithm to find shortest paths from start to all vertices
    """
    # Initialize distances with infinity
    distances = [float('inf')] * n
    distances[start] = 0
    
    # Priority queue: (distance, vertex)
    pq = [(0, start)]
    visited = set()
    
    while pq:
        current_dist, u = heapq.heappop(pq)
        
        # Skip if already visited
        if u in visited:
            continue
        
        visited.add(u)
        
        # Check all neighbors
        for v, weight in graph[u]:
            if v not in visited:
                new_dist = current_dist + weight
                if new_dist < distances[v]:
                    distances[v] = new_dist
                    heapq.heappush(pq, (new_dist, v))
    
    return distances

def dijkstra_with_path(graph, start, end, n):
    """
    Dijkstra's algorithm that also returns the shortest path
    """
    distances = [float('inf')] * n
    distances[start] = 0
    previous = [-1] * n
    
    pq = [(0, start)]
    visited = set()
    
    while pq:
        current_dist, u = heapq.heappop(pq)
        
        if u in visited:
            continue
        
        visited.add(u)
        
        if u == end:
            break  # Found shortest path to end
        
        for v, weight in graph[u]:
            if v not in visited:
                new_dist = current_dist + weight
                if new_dist < distances[v]:
                    distances[v] = new_dist
                    previous[v] = u
                    heapq.heappush(pq, (new_dist, v))
    
    # Reconstruct path
    path = []
    current = end
    while current != -1:
        path.append(current)
        current = previous[current]
    
    path.reverse()
    
    return distances[end] if distances[end] != float('inf') else -1, path
```

## Practice Questions

1. [Network Delay Time](https://leetcode.com/problems/network-delay-time/) - Medium
2. [Path with Maximum Probability](https://leetcode.com/problems/path-with-maximum-probability/) - Medium
3. [Cheapest Flights Within K Stops](https://leetcode.com/problems/cheapest-flights-within-k-stops/) - Medium
4. [Minimum Path Sum](https://leetcode.com/problems/minimum-path-sum/) - Medium
5. [Dijkstra's Algorithm](https://practice.geeksforgeeks.org/problems/implementing-dijkstras-algorithm/1) - Medium
6. [Path with Minimum Effort](https://leetcode.com/problems/path-with-minimum-effort/) - Medium
7. [Swim in Rising Water](https://leetcode.com/problems/swim-in-rising-water/) - Hard
8. [Minimum Cost to Make at Least One Valid Path in a Grid](https://leetcode.com/problems/minimum-cost-to-make-at-least-one-valid-path-in-a-grid/) - Hard
9. [Maximum Probability Path](https://leetcode.com/problems/path-with-maximum-probability/) - Medium
10. [Find the City With the Smallest Number of Neighbors](https://leetcode.com/problems/find-the-city-with-the-smallest-number-of-neighbors-at-a-threshold-distance/) - Medium

## Notes

- Only works with non-negative edge weights
- Time complexity: O((V + E) log V) with binary heap
- Can be modified to find maximum probability instead of minimum distance

---

**Tags:** #dsa #dijkstra #leetcode

**Last Reviewed:** 2025-11-05