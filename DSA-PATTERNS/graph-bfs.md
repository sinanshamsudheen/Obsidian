## 🧠 Graph BFS

**Graph BFS (Breadth-First Search)** explores a graph level by level, visiting all neighbors before moving to the next level.

• Uses a queue to keep track of nodes to visit next
• Explores nodes in order of their distance from the start node
• Guarantees shortest path in unweighted graphs
• Time complexity: O(V + E), Space complexity: O(V) where V = vertices, E = edges

**Related:** [[Graph]] [[Queue]] [[BFS]] [[Shortest Path]]

---

## 💡 Python Code Snippet

```python
from collections import deque, defaultdict

def bfs_traversal(graph, start):
    """
    Basic BFS traversal of graph
    Returns list of nodes in BFS order
    """
    visited = set()
    queue = deque([start])
    result = []
    
    visited.add(start)
    
    while queue:
        node = queue.popleft()
        result.append(node)
        
        # Visit all unvisited neighbors
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return result

def shortest_path_unweighted(graph, start, end):
    """
    Find shortest path in unweighted graph using BFS
    """
    if start == end:
        return [start]
    
    visited = set([start])
    queue = deque([(start, [start])])  # (node, path_to_node)
    
    while queue:
        node, path = queue.popleft()
        
        for neighbor in graph[node]:
            if neighbor == end:
                return path + [neighbor]
            
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    
    return []  # No path found

def min_distance(graph, start, end):
    """
    Find minimum distance between two nodes
    """
    if start == end:
        return 0
    
    visited = set([start])
    queue = deque([(start, 0)])  # (node, distance)
    
    while queue:
        node, distance = queue.popleft()
        
        for neighbor in graph[node]:
            if neighbor == end:
                return distance + 1
            
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))
    
    return -1  # No path exists

def level_order_graph_traversal(graph, start):
    """
    BFS with level information
    Returns nodes grouped by their level/distance from start
    """
    visited = set([start])
    queue = deque([start])
    levels = []
    
    while queue:
        level_size = len(queue)
        current_level = []
        
        for _ in range(level_size):
            node = queue.popleft()
            current_level.append(node)
            
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        levels.append(current_level)
    
    return levels

def is_bipartite(graph):
    """
    Check if graph is bipartite using BFS coloring
    """
    color = {}
    
    # Check each component (for disconnected graphs)
    for start in graph:
        if start in color:
            continue
        
        # BFS to color this component
        queue = deque([start])
        color[start] = 0
        
        while queue:
            node = queue.popleft()
            
            for neighbor in graph[node]:
                if neighbor not in color:
                    # Color with opposite color
                    color[neighbor] = 1 - color[node]
                    queue.append(neighbor)
                elif color[neighbor] == color[node]:
                    # Same color as current node - not bipartite
                    return False
    
    return True

def connected_components(graph):
    """
    Find all connected components using BFS
    """
    visited = set()
    components = []
    
    for node in graph:
        if node not in visited:
            # BFS to find this component
            component = []
            queue = deque([node])
            visited.add(node)
            
            while queue:
                current = queue.popleft()
                component.append(current)
                
                for neighbor in graph[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            components.append(component)
    
    return components
```

---

## 🔗 LeetCode Practice Problems (10)

- **Number of Islands** – https://leetcode.com/problems/number-of-islands/
- **Rotting Oranges** – https://leetcode.com/problems/rotting-oranges/
- **Word Ladder** – https://leetcode.com/problems/word-ladder/
- **Open the Lock** – https://leetcode.com/problems/open-the-lock/
- **Minimum Knight Moves** – https://leetcode.com/problems/minimum-knight-moves/
- **Is Graph Bipartite?** – https://leetcode.com/problems/is-graph-bipartite/
- **Shortest Path in Binary Matrix** – https://leetcode.com/problems/shortest-path-in-binary-matrix/
- **01 Matrix** – https://leetcode.com/problems/01-matrix/
- **Perfect Squares** – https://leetcode.com/problems/perfect-squares/
- **Snakes and Ladders** – https://leetcode.com/problems/snakes-and-ladders/

---

## 🧠 Flashcard (for spaced repetition)

What is the Graph BFS pattern? :: • Explores graph level by level using queue, visiting all neighbors before moving to next level • Guarantees shortest path in unweighted graphs with O(V + E) time complexity • Used for shortest path, connected components, and level-based graph problems [[graph-bfs]] 