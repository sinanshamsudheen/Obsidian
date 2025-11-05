# Island Pattern (Matrix DFS/BFS)

## Explanation

The Island Pattern solves problems involving connected components in 2D grids. It typically uses DFS or BFS to explore connected cells representing land ('1') or water ('0'). This pattern is used for counting islands, finding maximum area islands, surrounded regions, and other grid-based connectivity problems.

## Algorithm (Word-based)

- Iterate through each cell in the matrix
- When encountering an unvisited 'land' cell, start DFS/BFS
- Explore all connected land cells using 4-directional movement
- Mark visited cells to avoid reprocessing
- Count components or calculate area as required

## Code Template

```python
def count_islands(grid):
    """
    Count number of islands in 2D grid
    """
    if not grid or not grid[0]:
        return 0
    
    rows, cols = len(grid), len(grid[0])
    count = 0
    
    def dfs(r, c):
        # Check bounds and if it's land
        if (r < 0 or r >= rows or c < 0 or c >= cols or 
            grid[r][c] != '1'):
            return
        
        # Mark as visited by changing to '0' or using visited set
        grid[r][c] = '0'
        
        # Explore neighbors in 4 directions
        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                dfs(r, c)  # Explore entire island
                count += 1   # Increment island count
    
    return count

def max_area_of_island(grid):
    """
    Find maximum area of an island
    """
    if not grid:
        return 0
    
    rows, cols = len(grid), len(grid[0])
    max_area = 0
    
    def dfs(r, c):
        if (r < 0 or r >= rows or c < 0 or c >= cols or 
            grid[r][c] != 1):
            return 0
        
        grid[r][c] = 0  # Mark as visited
        area = 1
        
        # Explore neighbors and accumulate area
        area += dfs(r + 1, c)
        area += dfs(r - 1, c)
        area += dfs(r, c + 1)
        area += dfs(r, c - 1)
        
        return area
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                current_area = dfs(r, c)
                max_area = max(max_area, current_area)
    
    return max_area

def surrounded_regions(board):
    """
    Capture surrounded regions in matrix
    """
    if not board:
        return
    
    rows, cols = len(board), len(board[0])
    
    def dfs(r, c):
        if (r < 0 or r >= rows or c < 0 or c >= cols or 
            board[r][c] != 'O'):
            return
        
        board[r][c] = 'T'  # Temporarily mark as safe
        
        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)
    
    # Mark regions connected to borders
    for r in range(rows):
        for c in range(cols):
            if (board[r][c] == 'O' and 
                (r in [0, rows - 1] or c in [0, cols - 1])):
                dfs(r, c)
    
    # Flip remaining 'O' to 'X' and restore 'T' to 'O'
    for r in range(rows):
        for c in range(cols):
            if board[r][c] == 'O':
                board[r][c] = 'X'
            elif board[r][c] == 'T':
                board[r][c] = 'O'
```

## Practice Questions

1. [Number of Islands](https://leetcode.com/problems/number-of-islands/) - Medium
2. [Max Area of Island](https://leetcode.com/problems/max-area-of-island/) - Medium
3. [Count Sub Islands](https://leetcode.com/problems/count-sub-islands/) - Medium
4. [Surrounded Regions](https://leetcode.com/problems/surrounded-regions/) - Medium
5. [Number of Closed Islands](https://leetcode.com/problems/number-of-closed-islands/) - Medium
6. [Number of Distinct Islands](https://leetcode.com/problems/number-of-distinct-islands/) - Medium
7. [Number of Enclaves](https://leetcode.com/problems/number-of-enclaves/) - Medium
8. [Shortest Path in Binary Matrix](https://leetcode.com/problems/shortest-path-in-binary-matrix/) - Medium
9. [Pacific Atlantic Water Flow](https://leetcode.com/problems/pacific-atlantic-water-flow/) - Medium
10. [01 Matrix](https://leetcode.com/problems/01-matrix/) - Medium

## Notes

- DFS or BFS can be used depending on problem requirements
- 4-directional movement is most common (up, down, left, right)
- Can use external visited array instead of modifying input

---

**Tags:** #dsa #island-pattern #matrix-dfs #matrix-bfs #leetcode

**Last Reviewed:** 2025-11-05