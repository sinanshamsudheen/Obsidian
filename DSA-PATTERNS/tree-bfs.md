## 🧠 Tree BFS

**Tree BFS (Breadth-First Search)** traverses a tree level by level from left to right using a queue data structure.

• Visits all nodes at current level before moving to next level
• Uses a queue to keep track of nodes to visit next
• Useful for level-order traversal, finding tree width, and shortest path problems
• Time complexity: O(n), Space complexity: O(w) where w is maximum width

**Related:** [[Tree]] [[Queue]] [[BFS]]

---

## 💡 Python Code Snippet

```python
from collections import deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def level_order_traversal(root):
    """
    Basic level-order traversal of tree
    Returns list of values level by level
    """
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        current_level = []
        
        # Process all nodes at current level
        for _ in range(level_size):
            node = queue.popleft()
            current_level.append(node.val)
            
            # Add children to queue for next level
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(current_level)
    
    return result

def zigzag_level_order(root):
    """
    Zigzag level order traversal (alternate left-to-right and right-to-left)
    """
    if not root:
        return []
    
    result = []
    queue = deque([root])
    left_to_right = True
    
    while queue:
        level_size = len(queue)
        current_level = []
        
        for _ in range(level_size):
            node = queue.popleft()
            current_level.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        # Reverse level if going right to left
        if not left_to_right:
            current_level.reverse()
        
        result.append(current_level)
        left_to_right = not left_to_right
    
    return result

def find_minimum_depth(root):
    """
    Find minimum depth (shortest path to leaf) using BFS
    """
    if not root:
        return 0
    
    queue = deque([(root, 1)])  # (node, depth)
    
    while queue:
        node, depth = queue.popleft()
        
        # If this is a leaf node, return its depth
        if not node.left and not node.right:
            return depth
        
        # Add children with incremented depth
        if node.left:
            queue.append((node.left, depth + 1))
        if node.right:
            queue.append((node.right, depth + 1))
    
    return 0

def level_averages(root):
    """
    Calculate average of each level in the tree
    """
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        level_sum = 0
        
        for _ in range(level_size):
            node = queue.popleft()
            level_sum += node.val
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(level_sum / level_size)
    
    return result
```

---

## 🔗 LeetCode Practice Problems (10)

- **Binary Tree Level Order Traversal** – https://leetcode.com/problems/binary-tree-level-order-traversal/
- **Binary Tree Level Order Traversal II** – https://leetcode.com/problems/binary-tree-level-order-traversal-ii/
- **Binary Tree Zigzag Level Order Traversal** – https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/
- **Minimum Depth of Binary Tree** – https://leetcode.com/problems/minimum-depth-of-binary-tree/
- **Maximum Depth of Binary Tree** – https://leetcode.com/problems/maximum-depth-of-binary-tree/
- **Average of Levels in Binary Tree** – https://leetcode.com/problems/average-of-levels-in-binary-tree/
- **Binary Tree Right Side View** – https://leetcode.com/problems/binary-tree-right-side-view/
- **Find Largest Value in Each Tree Row** – https://leetcode.com/problems/find-largest-value-in-each-tree-row/
- **Populating Next Right Pointers in Each Node** – https://leetcode.com/problems/populating-next-right-pointers-in-each-node/
- **Binary Tree Level Order Traversal** – https://leetcode.com/problems/binary-tree-level-order-traversal/

---

## 🧠 Flashcard (for spaced repetition)

What is the Tree BFS pattern? ::

• Traverses tree level by level using queue data structure to visit all nodes at current level first
• Time O(n), Space O(w) where w is maximum width of tree
• Used for level-order traversal, finding minimum depth, and level-based tree problems

[[tree-bfs]] 