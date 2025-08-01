## 🧠 Floyd-Warshall / All-Pairs Shortest Path

**Floyd-Warshall** finds shortest paths between all pairs of vertices in a weighted graph, handling negative edges but not negative cycles.

• Dynamic programming approach that considers each vertex as intermediate point
• Works with negative edge weights but detects negative cycles
• Time complexity: O(V³), Space complexity: O(V²)
• Useful for dense graphs and when you need all-pairs shortest paths

**Related:** [[Graph]] [[Dynamic Programming]] [[Shortest Path]] [[Matrix]]

---

## 💡 Python Code Snippet

```python
def floyd_warshall(graph):
    """
    Classic Floyd-Warshall algorithm for all-pairs shortest paths
    graph: adjacency matrix where graph[i][j] = weight of edge i->j
    Use float('inf') for no edge
    """
    n = len(graph)
    # Initialize distance matrix
    dist = [[float('inf')] * n for _ in range(n)]
    
    # Copy graph and set diagonal to 0
    for i in range(n):
        for j in range(n):
            if i == j:
                dist[i][j] = 0
            elif graph[i][j] != 0:
                dist[i][j] = graph[i][j]
    
    # Floyd-Warshall: try each vertex as intermediate
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    
    return dist

def floyd_warshall_with_path_reconstruction(graph):
    """
    Floyd-Warshall with path reconstruction
    """
    n = len(graph)
    dist = [[float('inf')] * n for _ in range(n)]
    next_vertex = [[None] * n for _ in range(n)]
    
    # Initialize
    for i in range(n):
        for j in range(n):
            if i == j:
                dist[i][j] = 0
            elif graph[i][j] != 0:
                dist[i][j] = graph[i][j]
                next_vertex[i][j] = j
    
    # Floyd-Warshall with path tracking
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    next_vertex[i][j] = next_vertex[i][k]
    
    def get_path(start, end):
        if next_vertex[start][end] is None:
            return None
        
        path = [start]
        current = start
        while current != end:
            current = next_vertex[current][end]
            path.append(current)
        return path
    
    return dist, get_path

def detect_negative_cycle(graph):
    """
    Detect negative cycle using Floyd-Warshall
    """
    n = len(graph)
    dist = [[float('inf')] * n for _ in range(n)]
    
    # Initialize
    for i in range(n):
        for j in range(n):
            if i == j:
                dist[i][j] = 0
            elif graph[i][j] != 0:
                dist[i][j] = graph[i][j]
    
    # Floyd-Warshall
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    
    # Check for negative cycles (negative diagonal elements)
    for i in range(n):
        if dist[i][i] < 0:
            return True
    
    return False

def find_city_with_smallest_number_of_neighbors(n, edges, distance_threshold):
    """
    Find city with smallest number of reachable neighbors within threshold
    """
    # Build distance matrix
    dist = [[float('inf')] * n for _ in range(n)]
    
    for i in range(n):
        dist[i][i] = 0
    
    for u, v, w in edges:
        dist[u][v] = w
        dist[v][u] = w
    
    # Floyd-Warshall
    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    
    # Find city with minimum reachable neighbors
    min_neighbors = float('inf')
    result_city = 0
    
    for i in range(n):
        neighbors = sum(1 for j in range(n) if i != j and dist[i][j] <= distance_threshold)
        if neighbors <= min_neighbors:
            min_neighbors = neighbors
            result_city = i
    
    return result_city

def shortest_path_visiting_all_nodes(graph):
    """
    Shortest path visiting all nodes (TSP-like using Floyd-Warshall + DP)
    """
    n = len(graph)
    
    # Build distance matrix
    dist = [[float('inf')] * n for _ in range(n)]
    
    for i in range(n):
        dist[i][i] = 0
        for j in graph[i]:
            dist[i][j] = 1
    
    # Floyd-Warshall for all-pairs shortest paths
    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    
    # DP for TSP: dp[mask][i] = min cost to visit nodes in mask ending at i
    dp = {}
    
    def solve(mask, pos):
        if mask == (1 << n) - 1:  # All nodes visited
            return 0
        
        if (mask, pos) in dp:
            return dp[(mask, pos)]
        
        result = float('inf')
        for next_node in range(n):
            if mask & (1 << next_node) == 0:  # Not visited
                new_mask = mask | (1 << next_node)
                result = min(result, dist[pos][next_node] + solve(new_mask, next_node))
        
        dp[(mask, pos)] = result
        return result
    
    # Try starting from each node
    min_path = float('inf')
    for start in range(n):
        min_path = min(min_path, solve(1 << start, start))
    
    return min_path

def count_paths_with_maximum_probability(n, edges, prob, start, end):
    """
    Find path with maximum probability (modified Floyd-Warshall)
    """
    # Initialize probability matrix
    max_prob = [[0.0] * n for _ in range(n)]
    
    for i in range(n):
        max_prob[i][i] = 1.0
    
    for i, (u, v) in enumerate(edges):
        max_prob[u][v] = prob[i]
        max_prob[v][u] = prob[i]
    
    # Modified Floyd-Warshall for maximum probability
    for k in range(n):
        for i in range(n):
            for j in range(n):
                max_prob[i][j] = max(max_prob[i][j], 
                                   max_prob[i][k] * max_prob[k][j])
    
    return max_prob[start][end]

def minimum_cost_to_make_connected(n, connections):
    """
    Minimum cost to make all cities connected
    """
    if len(connections) < n - 1:
        return -1  # Not enough edges
    
    # Build cost matrix
    cost = [[float('inf')] * n for _ in range(n)]
    
    for i in range(n):
        cost[i][i] = 0
    
    for u, v, c in connections:
        cost[u][v] = min(cost[u][v], c)
        cost[v][u] = min(cost[v][u], c)
    
    # Floyd-Warshall for minimum costs
    for k in range(n):
        for i in range(n):
            for j in range(n):
                cost[i][j] = min(cost[i][j], cost[i][k] + cost[k][j])
    
    # Use MST approach (Prim's algorithm)
    visited = [False] * n
    min_cost = [float('inf')] * n
    min_cost[0] = 0
    total_cost = 0
    
    for _ in range(n):
        u = -1
        for v in range(n):
            if not visited[v] and (u == -1 or min_cost[v] < min_cost[u]):
                u = v
        
        visited[u] = True
        total_cost += min_cost[u]
        
        for v in range(n):
            if not visited[v]:
                min_cost[v] = min(min_cost[v], cost[u][v])
    
    return total_cost

def network_delay_time_floyd_warshall(times, n, k):
    """
    Network delay time using Floyd-Warshall
    """
    # Build distance matrix
    dist = [[float('inf')] * n for _ in range(n)]
    
    for i in range(n):
        dist[i][i] = 0
    
    for u, v, w in times:
        dist[u-1][v-1] = w  # Convert to 0-indexed
    
    # Floyd-Warshall
    for mid in range(n):
        for i in range(n):
            for j in range(n):
                dist[i][j] = min(dist[i][j], dist[i][mid] + dist[mid][j])
    
    # Find maximum distance from source k-1
    max_dist = 0
    for i in range(n):
        if dist[k-1][i] == float('inf'):
            return -1
        max_dist = max(max_dist, dist[k-1][i])
    
    return max_dist

def floyd_warshall_transitive_closure(graph):
    """
    Compute transitive closure using Floyd-Warshall
    """
    n = len(graph)
    reach = [[False] * n for _ in range(n)]
    
    # Initialize reachability matrix
    for i in range(n):
        for j in range(n):
            reach[i][j] = (i == j) or graph[i][j]
    
    # Floyd-Warshall for transitive closure
    for k in range(n):
        for i in range(n):
            for j in range(n):
                reach[i][j] = reach[i][j] or (reach[i][k] and reach[k][j])
    
    return reach

def count_negative_cycles(graph):
    """
    Count number of negative cycles in graph
    """
    n = len(graph)
    dist = [[float('inf')] * n for _ in range(n)]
    
    # Initialize
    for i in range(n):
        for j in range(n):
            if i == j:
                dist[i][j] = 0
            elif graph[i][j] != 0:
                dist[i][j] = graph[i][j]
    
    # Floyd-Warshall
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] != float('inf') and dist[k][j] != float('inf'):
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    
    # Count negative cycles
    negative_cycles = 0
    for i in range(n):
        if dist[i][i] < 0:
            negative_cycles += 1
    
    return negative_cycles

def shortest_path_with_alternating_colors(n, red_edges, blue_edges):
    """
    Shortest path with alternating edge colors
    """
    # State: (node, last_color) where 0=red, 1=blue
    # Build adjacency lists
    red_adj = [[] for _ in range(n)]
    blue_adj = [[] for _ in range(n)]
    
    for u, v in red_edges:
        red_adj[u].append(v)
    
    for u, v in blue_edges:
        blue_adj[u].append(v)
    
    # Distance matrix: dist[node][color] = shortest distance ending with color
    dist = [[float('inf')] * 2 for _ in range(n)]
    dist[0][0] = dist[0][1] = 0  # Can start with either color
    
    # BFS-like approach
    from collections import deque
    queue = deque([(0, 0, 0), (0, 1, 0)])  # (node, last_color, distance)
    
    while queue:
        node, last_color, d = queue.popleft()
        
        if d > dist[node][last_color]:
            continue
        
        # Try opposite color edges
        adj_list = blue_adj if last_color == 0 else red_adj
        new_color = 1 - last_color
        
        for next_node in adj_list[node]:
            if d + 1 < dist[next_node][new_color]:
                dist[next_node][new_color] = d + 1
                queue.append((next_node, new_color, d + 1))
    
    result = []
    for i in range(n):
        min_dist = min(dist[i][0], dist[i][1])
        result.append(-1 if min_dist == float('inf') else min_dist)
    
    return result
```

---

## 🔗 LeetCode Practice Problems (10)

- **Find the City With the Smallest Number of Neighbors at a Threshold Distance** – https://leetcode.com/problems/find-the-city-with-the-smallest-number-of-neighbors-at-a-threshold-distance/
- **Network Delay Time** – https://leetcode.com/problems/network-delay-time/
- **Shortest Path Visiting All Nodes** – https://leetcode.com/problems/shortest-path-visiting-all-nodes/
- **Path With Maximum Probability** – https://leetcode.com/problems/path-with-maximum-probability/
- **Minimum Cost to Make at Least One Valid Path in a Grid** – https://leetcode.com/problems/minimum-cost-to-make-at-least-one-valid-path-in-a-grid/
- **Shortest Path with Alternating Colors** – https://leetcode.com/problems/shortest-path-with-alternating-colors/
- **Find Critical and Pseudo-Critical Edges in MST** – https://leetcode.com/problems/find-critical-and-pseudo-critical-edges-in-minimum-spanning-tree/
- **Number of Ways to Arrive at Destination** – https://leetcode.com/problems/number-of-ways-to-arrive-at-destination/
- **Minimum Cost to Connect All Points** – https://leetcode.com/problems/minimum-cost-to-connect-all-points/
- **Reachable Nodes In Subdivided Graph** – https://leetcode.com/problems/reachable-nodes-in-subdivided-graph/

---

## 🧠 Flashcard (for spaced repetition)

What is the Floyd-Warshall / All-Pairs Shortest Path pattern? ::

• Dynamic programming algorithm finding shortest paths between all vertex pairs in O(V³) time
• Handles negative edges but detects negative cycles, considers each vertex as intermediate point
• Used for dense graphs, transitive closure, and when all-pairs shortest paths are needed

[[floyd-warshall-all-pairs-shortest-path]] 