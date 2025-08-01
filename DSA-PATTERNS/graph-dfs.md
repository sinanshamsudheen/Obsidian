## 🧠 Graph DFS

**Graph DFS (Depth-First Search)** explores a graph by going as deep as possible along each branch before backtracking.

• Uses recursion (implicit stack) or explicit stack for traversal
• Explores one path completely before trying alternative paths
• Useful for cycle detection, topological sorting, and strongly connected components
• Time complexity: O(V + E), Space complexity: O(V) where V = vertices, E = edges

**Related:** [[Graph]] [[Stack]] [[DFS]] [[Recursion]]

---

## 💡 Python Code Snippet

```python
from collections import defaultdict

def dfs_recursive(graph, start, visited=None):
    """
    Recursive DFS traversal
    """
    if visited is None:
        visited = set()
    
    visited.add(start)
    result = [start]
    
    for neighbor in graph[start]:
        if neighbor not in visited:
            result.extend(dfs_recursive(graph, neighbor, visited))
    
    return result

def dfs_iterative(graph, start):
    """
    Iterative DFS using explicit stack
    """
    visited = set()
    stack = [start]
    result = []
    
    while stack:
        node = stack.pop()
        
        if node not in visited:
            visited.add(node)
            result.append(node)
            
            # Add neighbors to stack (reverse order for left-to-right traversal)
            for neighbor in reversed(graph[node]):
                if neighbor not in visited:
                    stack.append(neighbor)
    
    return result

def has_cycle_undirected(graph):
    """
    Detect cycle in undirected graph using DFS
    """
    visited = set()
    
    def dfs(node, parent):
        visited.add(node)
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                if dfs(neighbor, node):
                    return True
            elif neighbor != parent:
                # Back edge found (cycle detected)
                return True
        
        return False
    
    # Check each component
    for node in graph:
        if node not in visited:
            if dfs(node, -1):
                return True
    
    return False

def has_cycle_directed(graph):
    """
    Detect cycle in directed graph using DFS with coloring
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    color = defaultdict(int)
    
    def dfs(node):
        if color[node] == GRAY:
            return True  # Back edge (cycle found)
        if color[node] == BLACK:
            return False  # Already processed
        
        color[node] = GRAY  # Mark as being processed
        
        for neighbor in graph[node]:
            if dfs(neighbor):
                return True
        
        color[node] = BLACK  # Mark as completely processed
        return False
    
    for node in graph:
        if color[node] == WHITE:
            if dfs(node):
                return True
    
    return False

def find_path(graph, start, end, path=None):
    """
    Find a path between two nodes using DFS
    """
    if path is None:
        path = []
    
    path = path + [start]
    
    if start == end:
        return path
    
    for neighbor in graph[start]:
        if neighbor not in path:  # Avoid cycles
            new_path = find_path(graph, neighbor, end, path)
            if new_path:
                return new_path
    
    return None

def all_paths(graph, start, end, path=None, all_paths_list=None):
    """
    Find all paths between two nodes
    """
    if path is None:
        path = []
    if all_paths_list is None:
        all_paths_list = []
    
    path = path + [start]
    
    if start == end:
        all_paths_list.append(path)
        return all_paths_list
    
    for neighbor in graph[start]:
        if neighbor not in path:
            all_paths(graph, neighbor, end, path, all_paths_list)
    
    return all_paths_list

def strongly_connected_components(graph):
    """
    Find strongly connected components using Kosaraju's algorithm
    """
    def dfs1(node, visited, stack):
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs1(neighbor, visited, stack)
        stack.append(node)
    
    def dfs2(node, visited, component, transposed_graph):
        visited.add(node)
        component.append(node)
        for neighbor in transposed_graph[node]:
            if neighbor not in visited:
                dfs2(neighbor, visited, component, transposed_graph)
    
    # Step 1: Fill stack with finish times
    visited = set()
    stack = []
    for node in graph:
        if node not in visited:
            dfs1(node, visited, stack)
    
    # Step 2: Create transposed graph
    transposed_graph = defaultdict(list)
    for node in graph:
        for neighbor in graph[node]:
            transposed_graph[neighbor].append(node)
    
    # Step 3: Process nodes in reverse finish time order
    visited = set()
    sccs = []
    
    while stack:
        node = stack.pop()
        if node not in visited:
            component = []
            dfs2(node, visited, component, transposed_graph)
            sccs.append(component)
    
    return sccs
```

---

## 🔗 LeetCode Practice Problems (10)

- **Clone Graph** – https://leetcode.com/problems/clone-graph/
- **Number of Islands** – https://leetcode.com/problems/number-of-islands/
- **Course Schedule** – https://leetcode.com/problems/course-schedule/
- **Course Schedule II** – https://leetcode.com/problems/course-schedule-ii/
- **Pacific Atlantic Water Flow** – https://leetcode.com/problems/pacific-atlantic-water-flow/
- **Number of Connected Components in an Undirected Graph** – https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/
- **Graph Valid Tree** – https://leetcode.com/problems/graph-valid-tree/
- **Surrounded Regions** – https://leetcode.com/problems/surrounded-regions/
- **Word Search** – https://leetcode.com/problems/word-search/
- **Reconstruct Itinerary** – https://leetcode.com/problems/reconstruct-itinerary/

---

## 🧠 Flashcard (for spaced repetition)

What is the Graph DFS pattern? :: • Explores graph by going deep along each branch before backtracking using recursion or stack • Time O(V + E), Space O(V); used for cycle detection, topological sorting, and path finding • Goes deep first unlike BFS which explores level by level [[graph-dfs]] 