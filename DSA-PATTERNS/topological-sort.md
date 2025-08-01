## 🧠 Topological Sort

**Topological Sort** arranges vertices of a directed acyclic graph (DAG) in a linear order where every directed edge goes from earlier to later in the ordering.

• Works only on directed acyclic graphs (DAGs)
• Two main approaches: DFS-based (using recursion stack) and BFS-based (Kahn's algorithm)
• Useful for scheduling tasks, course prerequisites, and dependency resolution
• Time complexity: O(V + E), Space complexity: O(V)

**Related:** [[Graph]] [[DFS]] [[BFS]] [[DAG]]

---

## 💡 Python Code Snippet

```python
from collections import defaultdict, deque

def topological_sort_dfs(graph, num_vertices):
    """
    Topological sort using DFS approach
    """
    visited = [False] * num_vertices
    stack = []
    
    def dfs(vertex):
        visited[vertex] = True
        
        # Visit all neighbors
        for neighbor in graph[vertex]:
            if not visited[neighbor]:
                dfs(neighbor)
        
        # Push to stack after visiting all neighbors
        stack.append(vertex)
    
    # Call DFS for all unvisited vertices
    for vertex in range(num_vertices):
        if not visited[vertex]:
            dfs(vertex)
    
    # Return vertices in topological order
    return stack[::-1]

def topological_sort_kahns(graph, num_vertices):
    """
    Topological sort using Kahn's algorithm (BFS approach)
    """
    # Calculate in-degrees
    in_degree = [0] * num_vertices
    for vertex in range(num_vertices):
        for neighbor in graph[vertex]:
            in_degree[neighbor] += 1
    
    # Initialize queue with vertices having no incoming edges
    queue = deque()
    for vertex in range(num_vertices):
        if in_degree[vertex] == 0:
            queue.append(vertex)
    
    topo_order = []
    
    while queue:
        vertex = queue.popleft()
        topo_order.append(vertex)
        
        # Reduce in-degree of neighbors
        for neighbor in graph[vertex]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    # Check if graph has cycle
    if len(topo_order) != num_vertices:
        return []  # Cycle detected
    
    return topo_order

def can_finish_courses(num_courses, prerequisites):
    """
    Determine if all courses can be finished given prerequisites
    LeetCode: Course Schedule
    """
    # Build adjacency list
    graph = defaultdict(list)
    in_degree = [0] * num_courses
    
    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1
    
    # Kahn's algorithm
    queue = deque()
    for i in range(num_courses):
        if in_degree[i] == 0:
            queue.append(i)
    
    finished_courses = 0
    
    while queue:
        course = queue.popleft()
        finished_courses += 1
        
        for next_course in graph[course]:
            in_degree[next_course] -= 1
            if in_degree[next_course] == 0:
                queue.append(next_course)
    
    return finished_courses == num_courses

def find_course_order(num_courses, prerequisites):
    """
    Find order to finish all courses
    LeetCode: Course Schedule II
    """
    # Build graph
    graph = defaultdict(list)
    in_degree = [0] * num_courses
    
    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1
    
    # Topological sort using Kahn's algorithm
    queue = deque()
    for i in range(num_courses):
        if in_degree[i] == 0:
            queue.append(i)
    
    course_order = []
    
    while queue:
        course = queue.popleft()
        course_order.append(course)
        
        for next_course in graph[course]:
            in_degree[next_course] -= 1
            if in_degree[next_course] == 0:
                queue.append(next_course)
    
    return course_order if len(course_order) == num_courses else []

def alien_dictionary_order(words):
    """
    Find order of characters in alien dictionary
    """
    # Build graph from word comparisons
    graph = defaultdict(set)
    in_degree = defaultdict(int)
    
    # Initialize all characters
    for word in words:
        for char in word:
            in_degree[char] = 0
    
    # Build edges by comparing adjacent words
    for i in range(len(words) - 1):
        word1, word2 = words[i], words[i + 1]
        min_len = min(len(word1), len(word2))
        
        # Find first different character
        for j in range(min_len):
            if word1[j] != word2[j]:
                if word2[j] not in graph[word1[j]]:
                    graph[word1[j]].add(word2[j])
                    in_degree[word2[j]] += 1
                break
    
    # Topological sort
    queue = deque()
    for char in in_degree:
        if in_degree[char] == 0:
            queue.append(char)
    
    result = []
    while queue:
        char = queue.popleft()
        result.append(char)
        
        for neighbor in graph[char]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    return ''.join(result) if len(result) == len(in_degree) else ""
```

---

## 🔗 LeetCode Practice Problems (10)

- **Course Schedule** – https://leetcode.com/problems/course-schedule/
- **Course Schedule II** – https://leetcode.com/problems/course-schedule-ii/
- **Alien Dictionary** – https://leetcode.com/problems/alien-dictionary/
- **Minimum Height Trees** – https://leetcode.com/problems/minimum-height-trees/
- **Course Schedule III** – https://leetcode.com/problems/course-schedule-iii/
- **Parallel Courses** – https://leetcode.com/problems/parallel-courses/
- **Sort Items by Groups Respecting Dependencies** – https://leetcode.com/problems/sort-items-by-groups-respecting-dependencies/
- **Sequence Reconstruction** – https://leetcode.com/problems/sequence-reconstruction/
- **Build a Matrix With Conditions** – https://leetcode.com/problems/build-a-matrix-with-conditions/
- **Find All Possible Recipes from Given Supplies** – https://leetcode.com/problems/find-all-possible-recipes-from-given-supplies/

---

## 🧠 Flashcard (for spaced repetition)

What is the Topological Sort pattern? ::

• Arranges vertices of directed acyclic graph in linear order where edges go from earlier to later vertices
• Two approaches: DFS-based (recursion stack) and BFS-based (Kahn's algorithm using in-degrees)
• Used for task scheduling, course prerequisites, and dependency resolution problems

[[topological-sort]] 