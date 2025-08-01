## 🧠 Dijkstra's Algorithm

**Dijkstra's Algorithm** finds the shortest path from a source vertex to all other vertices in a weighted graph with non-negative edge weights.

• Greedy algorithm that always selects the unvisited vertex with minimum distance
• Uses priority queue (min-heap) for efficient vertex selection
• Time complexity: O((V + E) log V), Space complexity: O(V)
• Cannot handle negative edge weights but guarantees optimal solutions for non-negative weights

**Related:** [[Graph]] [[Heap]] [[Shortest Path]] [[Greedy]]

---

## 💡 Python Code Snippet

```python
import heapq
from collections import defaultdict

def dijkstra_basic(graph, start):
    """
    Basic Dijkstra's algorithm implementation
    graph: adjacency list {node: [(neighbor, weight), ...]}
    """
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    pq = [(0, start)]  # (distance, node)
    visited = set()
    
    while pq:
        current_dist, current = heapq.heappop(pq)
        
        if current in visited:
            continue
        
        visited.add(current)
        
        # Update distances to neighbors
        for neighbor, weight in graph[current]:
            distance = current_dist + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    
    return distances

def dijkstra_with_path(graph, start, end):
    """
    Dijkstra with path reconstruction
    """
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    previous = {}
    pq = [(0, start)]
    visited = set()
    
    while pq:
        current_dist, current = heapq.heappop(pq)
        
        if current == end:
            break
        
        if current in visited:
            continue
        
        visited.add(current)
        
        for neighbor, weight in graph[current]:
            distance = current_dist + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous[neighbor] = current
                heapq.heappush(pq, (distance, neighbor))
    
    # Reconstruct path
    path = []
    current = end
    while current in previous:
        path.append(current)
        current = previous[current]
    
    if current == start:
        path.append(start)
        path.reverse()
        return distances[end], path
    else:
        return float('inf'), []

def network_delay_time(times, n, k):
    """
    Network delay time using Dijkstra
    """
    # Build adjacency list
    graph = defaultdict(list)
    for u, v, w in times:
        graph[u].append((v, w))
    
    # Dijkstra from source k
    distances = {i: float('inf') for i in range(1, n + 1)}
    distances[k] = 0
    pq = [(0, k)]
    
    while pq:
        current_dist, current = heapq.heappop(pq)
        
        if current_dist > distances[current]:
            continue
        
        for neighbor, weight in graph[current]:
            distance = current_dist + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    
    # Find maximum distance
    max_time = max(distances.values())
    return max_time if max_time != float('inf') else -1

def cheapest_flights_k_stops(n, flights, src, dst, k):
    """
    Cheapest flights within k stops (modified Dijkstra)
    """
    # Build adjacency list
    graph = defaultdict(list)
    for u, v, price in flights:
        graph[u].append((v, price))
    
    # (cost, node, stops_used)
    pq = [(0, src, 0)]
    # Track minimum cost to reach each node with specific stops
    visited = {}
    
    while pq:
        cost, node, stops = heapq.heappop(pq)
        
        if node == dst:
            return cost
        
        if stops > k:
            continue
        
        # Skip if we've found better path to this node with same or fewer stops
        if (node, stops) in visited:
            continue
        
        visited[(node, stops)] = cost
        
        for neighbor, price in graph[node]:
            new_cost = cost + price
            new_stops = stops + 1
            
            # Only explore if we haven't exceeded stop limit
            if new_stops <= k + 1:
                heapq.heappush(pq, (new_cost, neighbor, new_stops))
    
    return -1

def path_with_maximum_probability(n, edges, prob, start, end):
    """
    Path with maximum probability (modified Dijkstra for maximization)
    """
    # Build adjacency list
    graph = defaultdict(list)
    for i, (u, v) in enumerate(edges):
        graph[u].append((v, prob[i]))
        graph[v].append((u, prob[i]))
    
    # Use negative probabilities for min-heap (to maximize probability)
    max_prob = [0.0] * n
    max_prob[start] = 1.0
    pq = [(-1.0, start)]  # (negative_probability, node)
    
    while pq:
        current_prob, current = heapq.heappop(pq)
        current_prob = -current_prob
        
        if current == end:
            return current_prob
        
        if current_prob < max_prob[current]:
            continue
        
        for neighbor, edge_prob in graph[current]:
            new_prob = current_prob * edge_prob
            
            if new_prob > max_prob[neighbor]:
                max_prob[neighbor] = new_prob
                heapq.heappush(pq, (-new_prob, neighbor))
    
    return 0.0

def minimum_effort_path(heights):
    """
    Path with minimum maximum effort using Dijkstra
    """
    rows, cols = len(heights), len(heights[0])
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    # effort[r][c] = minimum effort to reach (r, c)
    effort = [[float('inf')] * cols for _ in range(rows)]
    effort[0][0] = 0
    
    pq = [(0, 0, 0)]  # (effort, row, col)
    
    while pq:
        current_effort, row, col = heapq.heappop(pq)
        
        if row == rows - 1 and col == cols - 1:
            return current_effort
        
        if current_effort > effort[row][col]:
            continue
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            
            if 0 <= new_row < rows and 0 <= new_col < cols:
                new_effort = max(current_effort, 
                               abs(heights[new_row][new_col] - heights[row][col]))
                
                if new_effort < effort[new_row][new_col]:
                    effort[new_row][new_col] = new_effort
                    heapq.heappush(pq, (new_effort, new_row, new_col))
    
    return 0

def swim_in_rising_water(grid):
    """
    Swim in rising water using Dijkstra
    """
    n = len(grid)
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    # time[r][c] = minimum time to reach (r, c)
    time = [[float('inf')] * n for _ in range(n)]
    time[0][0] = grid[0][0]
    
    pq = [(grid[0][0], 0, 0)]  # (time, row, col)
    
    while pq:
        current_time, row, col = heapq.heappop(pq)
        
        if row == n - 1 and col == n - 1:
            return current_time
        
        if current_time > time[row][col]:
            continue
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            
            if 0 <= new_row < n and 0 <= new_col < n:
                new_time = max(current_time, grid[new_row][new_col])
                
                if new_time < time[new_row][new_col]:
                    time[new_row][new_col] = new_time
                    heapq.heappush(pq, (new_time, new_row, new_col))
    
    return -1

def shortest_path_binary_matrix(grid):
    """
    Shortest path in binary matrix using Dijkstra/BFS
    """
    n = len(grid)
    
    if grid[0][0] == 1 or grid[n-1][n-1] == 1:
        return -1
    
    if n == 1:
        return 1
    
    directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    
    # Since all edges have weight 1, BFS is sufficient
    from collections import deque
    queue = deque([(1, 0, 0)])  # (distance, row, col)
    visited = set([(0, 0)])
    
    while queue:
        dist, row, col = queue.popleft()
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            
            if (0 <= new_row < n and 0 <= new_col < n and 
                grid[new_row][new_col] == 0 and 
                (new_row, new_col) not in visited):
                
                if new_row == n - 1 and new_col == n - 1:
                    return dist + 1
                
                visited.add((new_row, new_col))
                queue.append((dist + 1, new_row, new_col))
    
    return -1

def find_shortest_path_all_keys(grid):
    """
    Shortest path to collect all keys using Dijkstra with state
    """
    rows, cols = len(grid), len(grid[0])
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    # Find start position and count keys
    start_row = start_col = 0
    key_count = 0
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '@':
                start_row, start_col = r, c
            elif grid[r][c].islower():
                key_count += 1
    
    # State: (row, col, keys_collected_bitmask)
    target_state = (1 << key_count) - 1  # All keys collected
    
    # BFS with state
    from collections import deque
    queue = deque([(0, start_row, start_col, 0)])  # (steps, row, col, keys)
    visited = set([(start_row, start_col, 0)])
    
    while queue:
        steps, row, col, keys = queue.popleft()
        
        if keys == target_state:
            return steps
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            
            if (0 <= new_row < rows and 0 <= new_col < cols and 
                grid[new_row][new_col] != '#'):
                
                cell = grid[new_row][new_col]
                new_keys = keys
                
                # Check if we can pass through this cell
                if cell.isupper():  # Lock
                    key_bit = 1 << (ord(cell.lower()) - ord('a'))
                    if not (keys & key_bit):  # Don't have the key
                        continue
                elif cell.islower():  # Key
                    key_bit = 1 << (ord(cell) - ord('a'))
                    new_keys |= key_bit
                
                state = (new_row, new_col, new_keys)
                if state not in visited:
                    visited.add(state)
                    queue.append((steps + 1, new_row, new_col, new_keys))
    
    return -1

def min_cost_connect_points(points):
    """
    Minimum cost to connect all points using Prim's algorithm (similar to Dijkstra)
    """
    n = len(points)
    
    def manhattan_distance(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
    
    # Start from point 0
    visited = set([0])
    pq = []
    
    # Add all edges from point 0
    for i in range(1, n):
        dist = manhattan_distance(points[0], points[i])
        heapq.heappush(pq, (dist, i))
    
    total_cost = 0
    
    while pq and len(visited) < n:
        cost, point = heapq.heappop(pq)
        
        if point in visited:
            continue
        
        visited.add(point)
        total_cost += cost
        
        # Add edges from newly visited point
        for i in range(n):
            if i not in visited:
                dist = manhattan_distance(points[point], points[i])
                heapq.heappush(pq, (dist, i))
    
    return total_cost

def dijkstra_with_modifications(graph, start, forbidden_nodes):
    """
    Dijkstra with forbidden nodes
    """
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    pq = [(0, start)]
    visited = set()
    
    while pq:
        current_dist, current = heapq.heappop(pq)
        
        if current in visited or current in forbidden_nodes:
            continue
        
        visited.add(current)
        
        for neighbor, weight in graph[current]:
            if neighbor not in forbidden_nodes:
                distance = current_dist + weight
                
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(pq, (distance, neighbor))
    
    return distances
```

---

## 🔗 LeetCode Practice Problems (10)

- **Network Delay Time** – https://leetcode.com/problems/network-delay-time/
- **Cheapest Flights Within K Stops** – https://leetcode.com/problems/cheapest-flights-within-k-stops/
- **Path With Maximum Probability** – https://leetcode.com/problems/path-with-maximum-probability/
- **Path with Minimum Effort** – https://leetcode.com/problems/path-with-minimum-effort/
- **Swim in Rising Water** – https://leetcode.com/problems/swim-in-rising-water/
- **Shortest Path in Binary Matrix** – https://leetcode.com/problems/shortest-path-in-binary-matrix/
- **Shortest Path to Get All Keys** – https://leetcode.com/problems/shortest-path-to-get-all-keys/
- **Min Cost to Connect All Points** – https://leetcode.com/problems/min-cost-to-connect-all-points/
- **The Maze II** – https://leetcode.com/problems/the-maze-ii/
- **Reachable Nodes In Subdivided Graph** – https://leetcode.com/problems/reachable-nodes-in-subdivided-graph/

---

## 🧠 Flashcard (for spaced repetition)

What is the Dijkstra's Algorithm pattern? :: • Greedy algorithm finding shortest paths from source to all vertices using priority queue (min-heap) • O((V + E) log V) time complexity, works only with non-negative edge weights • Applied to shortest path problems, network routing, and optimization with distance/cost metrics [[dijkstras-algorithm]] 