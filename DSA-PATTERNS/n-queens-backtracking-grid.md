## 🧠 N-Queens / Backtracking Grid

**N-Queens / Backtracking Grid** solves constraint satisfaction problems on grids using backtracking to place items while satisfying multiple constraints.

• Classic problem: place N queens on N×N board such that no two queens attack each other
• Uses backtracking to try placements row by row and backtrack when constraints violated
• Maintains constraint checking (rows, columns, diagonals for N-Queens)
• Generalizes to any grid-based constraint satisfaction problem

**Related:** [[Backtracking]] [[Grid]] [[Constraint Satisfaction]] [[Recursion]]

---

## 💡 Python Code Snippet

```python
def solve_n_queens(n):
    """
    Solve N-Queens problem and return all solutions
    """
    result = []
    board = [['.' for _ in range(n)] for _ in range(n)]
    
    def is_safe(row, col):
        # Check column
        for i in range(row):
            if board[i][col] == 'Q':
                return False
        
        # Check upper-left diagonal
        i, j = row - 1, col - 1
        while i >= 0 and j >= 0:
            if board[i][j] == 'Q':
                return False
            i -= 1
            j -= 1
        
        # Check upper-right diagonal
        i, j = row - 1, col + 1
        while i >= 0 and j < n:
            if board[i][j] == 'Q':
                return False
            i -= 1
            j += 1
        
        return True
    
    def backtrack(row):
        if row == n:
            # Found a solution
            result.append([''.join(row) for row in board])
            return
        
        for col in range(n):
            if is_safe(row, col):
                board[row][col] = 'Q'
                backtrack(row + 1)
                board[row][col] = '.'  # Backtrack
    
    backtrack(0)
    return result

def total_n_queens(n):
    """
    Count total number of N-Queens solutions
    """
    def backtrack(row, cols, diag1, diag2):
        if row == n:
            return 1
        
        count = 0
        for col in range(n):
            # Check if position is safe using sets
            if col in cols or (row - col) in diag1 or (row + col) in diag2:
                continue
            
            # Place queen
            cols.add(col)
            diag1.add(row - col)
            diag2.add(row + col)
            
            count += backtrack(row + 1, cols, diag1, diag2)
            
            # Backtrack
            cols.remove(col)
            diag1.remove(row - col)
            diag2.remove(row + col)
        
        return count
    
    return backtrack(0, set(), set(), set())

def sudoku_solver(board):
    """
    Solve Sudoku puzzle using backtracking
    """
    def is_valid(row, col, num):
        # Check row
        for j in range(9):
            if board[row][j] == num:
                return False
        
        # Check column
        for i in range(9):
            if board[i][col] == num:
                return False
        
        # Check 3x3 box
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(start_row, start_row + 3):
            for j in range(start_col, start_col + 3):
                if board[i][j] == num:
                    return False
        
        return True
    
    def solve():
        for i in range(9):
            for j in range(9):
                if board[i][j] == '.':
                    for num in '123456789':
                        if is_valid(i, j, num):
                            board[i][j] = num
                            
                            if solve():
                                return True
                            
                            board[i][j] = '.'  # Backtrack
                    
                    return False  # No valid number found
        return True  # All cells filled
    
    solve()
    return board

def word_search_ii(board, words):
    """
    Find all words in board using backtracking with Trie
    """
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.word = None
    
    def build_trie(words):
        root = TrieNode()
        for word in words:
            node = root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.word = word
        return root
    
    def backtrack(row, col, parent):
        char = board[row][col]
        current_node = parent.children[char]
        
        # Found a word
        if current_node.word:
            result.add(current_node.word)
        
        # Mark cell as visited
        board[row][col] = '#'
        
        # Explore neighbors
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            
            if (0 <= new_row < len(board) and 0 <= new_col < len(board[0]) and
                board[new_row][new_col] in current_node.children):
                
                backtrack(new_row, new_col, current_node)
        
        # Restore cell
        board[row][col] = char
        
        # Optimization: remove leaf nodes
        if not current_node.children:
            parent.children.pop(char)
    
    root = build_trie(words)
    result = set()
    
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] in root.children:
                backtrack(i, j, root)
    
    return list(result)

def rat_in_maze(maze):
    """
    Find path for rat from top-left to bottom-right
    """
    n = len(maze)
    solution = [[0 for _ in range(n)] for _ in range(n)]
    
    def is_safe(row, col):
        return (0 <= row < n and 0 <= col < n and 
                maze[row][col] == 1 and solution[row][col] == 0)
    
    def solve_maze(row, col):
        # Reached destination
        if row == n - 1 and col == n - 1 and maze[row][col] == 1:
            solution[row][col] = 1
            return True
        
        if is_safe(row, col):
            # Mark current cell as part of solution
            solution[row][col] = 1
            
            # Try all four directions
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
            for dr, dc in directions:
                if solve_maze(row + dr, col + dc):
                    return True
            
            # Backtrack
            solution[row][col] = 0
            return False
        
        return False
    
    if solve_maze(0, 0):
        return solution
    else:
        return []

def knight_tour(n):
    """
    Find Knight's tour on n×n chessboard
    """
    # Knight moves
    moves = [(2, 1), (2, -1), (-2, 1), (-2, -1),
             (1, 2), (1, -2), (-1, 2), (-1, -2)]
    
    board = [[-1 for _ in range(n)] for _ in range(n)]
    
    def is_safe(row, col):
        return 0 <= row < n and 0 <= col < n and board[row][col] == -1
    
    def get_degree(row, col):
        # Count number of unvisited positions reachable from (row, col)
        count = 0
        for dr, dc in moves:
            if is_safe(row + dr, col + dc):
                count += 1
        return count
    
    def solve_knight_tour(row, col, move_count):
        board[row][col] = move_count
        
        if move_count == n * n - 1:
            return True
        
        # Get all possible next moves
        next_moves = []
        for dr, dc in moves:
            next_row, next_col = row + dr, col + dc
            if is_safe(next_row, next_col):
                degree = get_degree(next_row, next_col)
                next_moves.append((degree, next_row, next_col))
        
        # Sort by Warnsdorff's heuristic (visit squares with fewer onward moves first)
        next_moves.sort()
        
        for _, next_row, next_col in next_moves:
            if solve_knight_tour(next_row, next_col, move_count + 1):
                return True
        
        # Backtrack
        board[row][col] = -1
        return False
    
    if solve_knight_tour(0, 0, 0):
        return board
    else:
        return []

def coloring_graph(graph, colors):
    """
    Color graph such that no two adjacent vertices have same color
    """
    n = len(graph)
    color_assignment = [0] * n
    
    def is_safe(vertex, color):
        for neighbor in range(n):
            if graph[vertex][neighbor] and color_assignment[neighbor] == color:
                return False
        return True
    
    def solve_coloring(vertex):
        if vertex == n:
            return True
        
        for color in range(1, colors + 1):
            if is_safe(vertex, color):
                color_assignment[vertex] = color
                
                if solve_coloring(vertex + 1):
                    return True
                
                color_assignment[vertex] = 0  # Backtrack
        
        return False
    
    if solve_coloring(0):
        return color_assignment
    else:
        return []

def hamiltonian_path(graph):
    """
    Find Hamiltonian path (visits each vertex exactly once)
    """
    n = len(graph)
    path = []
    visited = [False] * n
    
    def solve_hamiltonian(vertex):
        path.append(vertex)
        visited[vertex] = True
        
        if len(path) == n:
            return True
        
        for next_vertex in range(n):
            if graph[vertex][next_vertex] and not visited[next_vertex]:
                if solve_hamiltonian(next_vertex):
                    return True
        
        # Backtrack
        path.pop()
        visited[vertex] = False
        return False
    
    # Try starting from each vertex
    for start in range(n):
        if solve_hamiltonian(start):
            return path
        path.clear()
        visited = [False] * n
    
    return []
```

---

## 🔗 LeetCode Practice Problems (10)

- **N-Queens** – https://leetcode.com/problems/n-queens/
- **N-Queens II** – https://leetcode.com/problems/n-queens-ii/
- **Sudoku Solver** – https://leetcode.com/problems/sudoku-solver/
- **Word Search II** – https://leetcode.com/problems/word-search-ii/
- **Word Search** – https://leetcode.com/problems/word-search/
- **Unique Paths III** – https://leetcode.com/problems/unique-paths-iii/
- **Robot Room Cleaner** – https://leetcode.com/problems/robot-room-cleaner/
- **Expression Add Operators** – https://leetcode.com/problems/expression-add-operators/
- **Partition to K Equal Sum Subsets** – https://leetcode.com/problems/partition-to-k-equal-sum-subsets/
- **Matchsticks to Square** – https://leetcode.com/problems/matchsticks-to-square/

---

## 🧠 Flashcard (for spaced repetition)

What is the N-Queens / Backtracking Grid pattern? :: • Solves constraint satisfaction problems on grids using backtracking with systematic placement and constraint checking • Places items row by row, validates constraints (like queen attacks), and backtracks when violated • Applied to N-Queens, Sudoku, graph coloring, and grid-based constraint problems [[n-queens-backtracking-grid]] 