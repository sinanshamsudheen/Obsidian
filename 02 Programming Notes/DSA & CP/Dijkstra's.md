
### created: 31-07-2025
---
## Problem Overview  

**Platform**: LeetCode / Codeforces / GFG / Other  
**Difficulty**: Easy / Medium / Hard  
**Link**: [Problem URL]()

---
### Concept  
What concept(s) does this problem test?  
(e.g. Binary Search, DP, Greedy, Graphs, Trees, etc.)

What is the key concept used in Dijkstra's::[e.g. Dynamic Programming with Memoization]

---
### Context  
Where and when is this useful?  
(e.g. Common in interviews, real-world problem class, etc.)

In what context does Dijkstra's apply::[Use-case explanation]

---
### Connection  
Which concepts/techniques/other problems are related?

- [[Binary Search]]
- [[Kadane’s Algorithm]]
- [[DP-State Optimization]]

What is Dijkstra's connected to::[[Linked_Problem_1]], [[Technique]]

---
### Concrete Example  
Explain how the solution works with a sample input/output.

```python
# Example solution
import heapq

def dijkstra(graph, start):
    n = len(graph)
    dist = [float('inf')] * n
    dist[start] = 0

    pq = [(0, start)]  # (distance, node)

    while pq:
        d, u = heapq.heappop(pq)

        if d > dist[u]:
            continue  # Already found a better path

        for v, weight in graph[u]:
            if dist[u] + weight < dist[v]:
                dist[v] = dist[u] + weight
                heapq.heappush(pq, (dist[v], v))

    return dist

# -----------------------------
# Step 1: Take input from user
# -----------------------------

n, m = map(int, input("Enter number of nodes and edges: ").split())
graph = [[] for _ in range(n)]

print("Enter each edge in format: u v w")
print("(0-indexed, undirected graph)")

for _ in range(m):
    u, v, w = map(int, input().split())
    graph[u].append((v, w))
    graph[v].append((u, w))  # Remove this for directed graph

# -----------------------------
# Step 2: Take source node
# -----------------------------

start = int(input("Enter source node: "))
distances = dijkstra(graph, start)

# -----------------------------
# Step 3: Print results
# -----------------------------

print(f"\nShortest distances from node {start}:")
for i, d in enumerate(distances):
    print(f"To node {i}: {d}")

```

Walk through a sample example of Dijkstra's::[Step-by-step explanation]


---
## Iterative Thinking

What confused me at first?
[Write it here]

What mistake did I make and how did I fix it?
[Write it here]

What would I ask in a follow-up or variation?
[Write here]

The most common mistake in Dijkstra's is {{[your insight]}}.


---
## Time & Space Complexity

Time: O(...)
Space: O(...)

What is the time and space complexity of Dijkstra's::Time: O(...), Space: O(...)


---
##### Tags

#dsa/Dijkstra's #cp #leetcode #gfg #interview #flashcard

