## 🧠 A* Search Algorithm

**A* Search Algorithm** is an informed search algorithm that finds the shortest path by using heuristics to guide the search towards the goal more efficiently than Dijkstra's algorithm.

• Combines actual cost from start (g-score) with heuristic estimate to goal (h-score)
• Uses f-score = g-score + h-score to prioritize nodes in exploration
• Guarantees optimal solution if heuristic is admissible (never overestimates)
• More efficient than Dijkstra when good heuristic is available

**Related:** [[Graph]] [[Heap]] [[Shortest Path]] [[Heuristic]] [[Search]]

---

## 💡 Python Code Snippet

```python
import heapq
from collections import defaultdict
import math

def a_star_grid(grid, start, goal):
    """
    A* search on 2D grid with Manhattan distance heuristic
    grid: 2D array where 0=free, 1=obstacle
    start, goal: (row, col) tuples
    """
    def heuristic(pos):
        """Manhattan distance heuristic"""
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    
    def euclidean_heuristic(pos):
        """Euclidean distance heuristic"""
        return math.sqrt((pos[0] - goal[0])**2 + (pos[1] - goal[1])**2)
    
    rows, cols = len(grid), len(grid[0])
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 4-directional
    
    # Priority queue: (f_score, g_score, position)
    open_set = [(heuristic(start), 0, start)]
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start)}
    visited = set()
    
    while open_set:
        current_f, current_g, current = heapq.heappop(open_set)
        
        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]
        
        if current in visited:
            continue
        
        visited.add(current)
        
        for dr, dc in directions:
            neighbor = (current[0] + dr, current[1] + dc)
            
            # Check bounds and obstacles
            if (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and 
                grid[neighbor[0]][neighbor[1]] == 0):
                
                tentative_g = g_score[current] + 1  # Cost of moving to neighbor
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor)
                    
                    if neighbor not in visited:
                        heapq.heappush(open_set, (f_score[neighbor], tentative_g, neighbor))
    
    return []  # No path found

def a_star_8_directional(grid, start, goal):
    """
    A* with 8-directional movement (including diagonals)
    """
    def heuristic(pos):
        # Diagonal distance heuristic
        dx = abs(pos[0] - goal[0])
        dy = abs(pos[1] - goal[1])
        return max(dx, dy)  # Chebyshev distance
    
    rows, cols = len(grid), len(grid[0])
    # 8 directions: 4 cardinal + 4 diagonal
    directions = [(0,1), (1,0), (0,-1), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]
    
    open_set = [(heuristic(start), 0, start)]
    came_from = {}
    g_score = {start: 0}
    visited = set()
    
    while open_set:
        current_f, current_g, current = heapq.heappop(open_set)
        
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]
        
        if current in visited:
            continue
        
        visited.add(current)
        
        for dr, dc in directions:
            neighbor = (current[0] + dr, current[1] + dc)
            
            if (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and 
                grid[neighbor[0]][neighbor[1]] == 0):
                
                # Diagonal moves cost sqrt(2), cardinal moves cost 1
                move_cost = math.sqrt(2) if abs(dr) + abs(dc) == 2 else 1
                tentative_g = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor)
                    
                    if neighbor not in visited:
                        heapq.heappush(open_set, (f_score, tentative_g, neighbor))
    
    return []

def a_star_weighted_graph(graph, start, goal, heuristic_func):
    """
    A* on weighted graph with custom heuristic
    graph: {node: [(neighbor, weight), ...]}
    heuristic_func: function that takes a node and returns heuristic value
    """
    open_set = [(heuristic_func(start), 0, start)]
    came_from = {}
    g_score = {start: 0}
    visited = set()
    
    while open_set:
        current_f, current_g, current = heapq.heappop(open_set)
        
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1], g_score[goal]
        
        if current in visited:
            continue
        
        visited.add(current)
        
        for neighbor, weight in graph.get(current, []):
            tentative_g = g_score[current] + weight
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic_func(neighbor)
                
                if neighbor not in visited:
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))
    
    return [], float('inf')

def shortest_path_visiting_all_nodes_astar(graph):
    """
    Shortest path visiting all nodes using A* with TSP heuristic
    """
    n = len(graph)
    
    def tsp_heuristic(state):
        """Minimum spanning tree heuristic for remaining nodes"""
        visited_mask, current_node = state
        unvisited = []
        
        for i in range(n):
            if not (visited_mask & (1 << i)):
                unvisited.append(i)
        
        if not unvisited:
            return 0
        
        # Simple heuristic: minimum edge to unvisited nodes
        min_cost = float('inf')
        for node in unvisited:
            for neighbor in range(n):
                if neighbor in graph[current_node]:
                    min_cost = min(min_cost, 1)  # Assuming unit weights
        
        return len(unvisited) - 1  # At least this many more edges needed
    
    # State: (visited_mask, current_node)
    start_states = [(1 << i, i) for i in range(n)]
    target_mask = (1 << n) - 1
    
    for start_mask, start_node in start_states:
        open_set = [(tsp_heuristic((start_mask, start_node)), 0, start_mask, start_node)]
        g_score = {(start_mask, start_node): 0}
        visited = set()
        
        while open_set:
            current_f, current_g, mask, node = heapq.heappop(open_set)
            
            if mask == target_mask:
                return current_g
            
            state = (mask, node)
            if state in visited:
                continue
            
            visited.add(state)
            
            # Try visiting each unvisited neighbor
            for neighbor in range(n):
                if not (mask & (1 << neighbor)) and neighbor in graph[node]:
                    new_mask = mask | (1 << neighbor)
                    new_state = (new_mask, neighbor)
                    tentative_g = current_g + 1  # Assuming unit weights
                    
                    if new_state not in g_score or tentative_g < g_score[new_state]:
                        g_score[new_state] = tentative_g
                        f_score = tentative_g + tsp_heuristic(new_state)
                        
                        if new_state not in visited:
                            heapq.heappush(open_set, (f_score, tentative_g, new_mask, neighbor))
    
    return float('inf')

def sliding_puzzle_astar(board):
    """
    Sliding puzzle solver using A* with Manhattan distance heuristic
    """
    def board_to_string(b):
        return ''.join(map(str, [cell for row in b for cell in row]))
    
    def string_to_board(s):
        return [[int(s[i*3 + j]) for j in range(3)] for i in range(2)]
    
    def manhattan_distance(board):
        """Manhattan distance heuristic"""
        distance = 0
        for i in range(2):
            for j in range(3):
                if board[i][j] != 0:
                    target_row = (board[i][j] - 1) // 3
                    target_col = (board[i][j] - 1) % 3
                    distance += abs(i - target_row) + abs(j - target_col)
        return distance
    
    def get_neighbors(board_str):
        """Get all possible moves from current state"""
        board = string_to_board(board_str)
        
        # Find empty cell (0)
        empty_row = empty_col = -1
        for i in range(2):
            for j in range(3):
                if board[i][j] == 0:
                    empty_row, empty_col = i, j
                    break
        
        neighbors = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        for dr, dc in directions:
            new_row, new_col = empty_row + dr, empty_col + dc
            
            if 0 <= new_row < 2 and 0 <= new_col < 3:
                # Swap empty cell with neighbor
                new_board = [row[:] for row in board]
                new_board[empty_row][empty_col] = board[new_row][new_col]
                new_board[new_row][new_col] = 0
                
                neighbors.append(board_to_string(new_board))
        
        return neighbors
    
    start = board_to_string(board)
    goal = "123450"
    
    if start == goal:
        return 0
    
    open_set = [(manhattan_distance(board), 0, start)]
    g_score = {start: 0}
    visited = set()
    
    while open_set:
        current_f, current_g, current = heapq.heappop(open_set)
        
        if current == goal:
            return current_g
        
        if current in visited:
            continue
        
        visited.add(current)
        
        for neighbor in get_neighbors(current):
            tentative_g = current_g + 1
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                neighbor_board = string_to_board(neighbor)
                f_score = tentative_g + manhattan_distance(neighbor_board)
                
                if neighbor not in visited:
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))
    
    return -1

def a_star_with_obstacles_and_costs(grid, start, goal, obstacle_penalty=float('inf')):
    """
    A* with different terrain costs
    grid values: 0=free(cost 1), 1=obstacle(infinite cost), 2=difficult(cost 3)
    """
    def heuristic(pos):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    
    def get_terrain_cost(cell_value):
        if cell_value == 0:
            return 1  # Normal terrain
        elif cell_value == 1:
            return obstacle_penalty  # Obstacle
        elif cell_value == 2:
            return 3  # Difficult terrain
        else:
            return 1  # Default
    
    rows, cols = len(grid), len(grid[0])
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    open_set = [(heuristic(start), 0, start)]
    came_from = {}
    g_score = {start: 0}
    visited = set()
    
    while open_set:
        current_f, current_g, current = heapq.heappop(open_set)
        
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1], g_score[goal]
        
        if current in visited:
            continue
        
        visited.add(current)
        
        for dr, dc in directions:
            neighbor = (current[0] + dr, current[1] + dc)
            
            if (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols):
                terrain_cost = get_terrain_cost(grid[neighbor[0]][neighbor[1]])
                
                if terrain_cost == obstacle_penalty:
                    continue  # Skip obstacles
                
                tentative_g = g_score[current] + terrain_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor)
                    
                    if neighbor not in visited:
                        heapq.heappush(open_set, (f_score, tentative_g, neighbor))
    
    return [], float('inf')

def a_star_multi_goal(grid, start, goals):
    """
    A* to find shortest path to any of multiple goals
    """
    def heuristic(pos):
        return min(abs(pos[0] - goal[0]) + abs(pos[1] - goal[1]) for goal in goals)
    
    rows, cols = len(grid), len(grid[0])
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    open_set = [(heuristic(start), 0, start)]
    came_from = {}
    g_score = {start: 0}
    visited = set()
    
    while open_set:
        current_f, current_g, current = heapq.heappop(open_set)
        
        if current in goals:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]
        
        if current in visited:
            continue
        
        visited.add(current)
        
        for dr, dc in directions:
            neighbor = (current[0] + dr, current[1] + dc)
            
            if (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and 
                grid[neighbor[0]][neighbor[1]] == 0):
                
                tentative_g = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor)
                    
                    if neighbor not in visited:
                        heapq.heappush(open_set, (f_score, tentative_g, neighbor))
    
    return []

def bidirectional_a_star(grid, start, goal):
    """
    Bidirectional A* search for potentially faster pathfinding
    """
    def heuristic_forward(pos):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    
    def heuristic_backward(pos):
        return abs(pos[0] - start[0]) + abs(pos[1] - start[1])
    
    rows, cols = len(grid), len(grid[0])
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    # Forward search
    open_forward = [(heuristic_forward(start), 0, start)]
    g_forward = {start: 0}
    came_from_forward = {}
    visited_forward = set()
    
    # Backward search
    open_backward = [(heuristic_backward(goal), 0, goal)]
    g_backward = {goal: 0}
    came_from_backward = {}
    visited_backward = set()
    
    while open_forward or open_backward:
        # Forward step
        if open_forward:
            current_f, current_g, current = heapq.heappop(open_forward)
            
            if current in visited_backward:
                # Found meeting point, reconstruct path
                path_forward = []
                node = current
                while node in came_from_forward:
                    path_forward.append(node)
                    node = came_from_forward[node]
                path_forward.append(start)
                path_forward.reverse()
                
                path_backward = []
                node = current
                while node in came_from_backward:
                    path_backward.append(node)
                    node = came_from_backward[node]
                path_backward.append(goal)
                
                return path_forward + path_backward[1:]
            
            visited_forward.add(current)
            
            for dr, dc in directions:
                neighbor = (current[0] + dr, current[1] + dc)
                
                if (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and 
                    grid[neighbor[0]][neighbor[1]] == 0 and 
                    neighbor not in visited_forward):
                    
                    tentative_g = g_forward[current] + 1
                    
                    if neighbor not in g_forward or tentative_g < g_forward[neighbor]:
                        came_from_forward[neighbor] = current
                        g_forward[neighbor] = tentative_g
                        f_score = tentative_g + heuristic_forward(neighbor)
                        heapq.heappush(open_forward, (f_score, tentative_g, neighbor))
        
        # Backward step
        if open_backward:
            current_f, current_g, current = heapq.heappop(open_backward)
            
            if current in visited_forward:
                # Found meeting point
                path_forward = []
                node = current
                while node in came_from_forward:
                    path_forward.append(node)
                    node = came_from_forward[node]
                path_forward.append(start)
                path_forward.reverse()
                
                path_backward = []
                node = current
                while node in came_from_backward:
                    path_backward.append(node)
                    node = came_from_backward[node]
                path_backward.append(goal)
                
                return path_forward + path_backward[1:]
            
            visited_backward.add(current)
            
            for dr, dc in directions:
                neighbor = (current[0] + dr, current[1] + dc)
                
                if (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and 
                    grid[neighbor[0]][neighbor[1]] == 0 and 
                    neighbor not in visited_backward):
                    
                    tentative_g = g_backward[current] + 1
                    
                    if neighbor not in g_backward or tentative_g < g_backward[neighbor]:
                        came_from_backward[neighbor] = current
                        g_backward[neighbor] = tentative_g
                        f_score = tentative_g + heuristic_backward(neighbor)
                        heapq.heappush(open_backward, (f_score, tentative_g, neighbor))
    
    return []
```

---

## 🔗 LeetCode Practice Problems (10)

- **Shortest Path in Binary Matrix** – https://leetcode.com/problems/shortest-path-in-binary-matrix/
- **Sliding Puzzle** – https://leetcode.com/problems/sliding-puzzle/
- **Minimum Cost to Make at Least One Valid Path in a Grid** – https://leetcode.com/problems/minimum-cost-to-make-at-least-one-valid-path-in-a-grid/
- **Shortest Path to Get All Keys** – https://leetcode.com/problems/shortest-path-to-get-all-keys/
- **Shortest Path Visiting All Nodes** – https://leetcode.com/problems/shortest-path-visiting-all-nodes/
- **Minimum Moves to Reach Target with Rotations** – https://leetcode.com/problems/minimum-moves-to-reach-target-with-rotations/
- **Cut Off Trees for Golf Event** – https://leetcode.com/problems/cut-off-trees-for-golf-event/
- **Escape a Large Maze** – https://leetcode.com/problems/escape-a-large-maze/
- **Minimum Moves to Move a Box to Their Target Location** – https://leetcode.com/problems/minimum-moves-to-move-a-box-to-their-target-location/
- **Race Car** – https://leetcode.com/problems/race-car/

---

## 🧠 Flashcard (for spaced repetition)

What is the A* Search Algorithm pattern? :: • Informed search using f-score = g-score + h-score where g=actual cost, h=heuristic estimate • More efficient than Dijkstra when good admissible heuristic available, guarantees optimal solution • Applied to pathfinding, puzzle solving, and optimization problems with spatial or domain knowledge [[a-star-search-algorithm]] 