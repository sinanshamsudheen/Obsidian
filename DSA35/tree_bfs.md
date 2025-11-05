# Tree Breadth First Search (BFS)

## Explanation

Tree BFS uses a queue to traverse a tree level by level, visiting nodes at each level before moving to the next. This pattern is useful for finding levels in a tree, finding shortest paths in unweighted trees, or when you need to process nodes in a layer-by-layer fashion. It typically uses a queue data structure for implementation.

## Algorithm (Word-based)

- Initialize queue with root node
- While queue is not empty
- Dequeue a node from the front
- Process the node
- Enqueue left and right children if they exist
- Continue until all nodes are processed

## Code Template

```python
from collections import deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def tree_bfs(root):
    """
    Breadth-first search traversal of a binary tree
    """
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        node = queue.popleft()
        result.append(node.val)
        
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    
    return result
```

## Practice Questions

1. [Binary Tree Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/) - Medium
2. [Binary Tree Zigzag Level Order Traversal](https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/) - Medium
3. [Binary Tree Right Side View](https://leetcode.com/problems/binary-tree-right-side-view/) - Medium
4. [Maximum Depth of Binary Tree](https://leetcode.com/problems/maximum-depth-of-binary-tree/) - Easy
5. [Minimum Depth of Binary Tree](https://leetcode.com/problems/minimum-depth-of-binary-tree/) - Easy
6. [Binary Tree Level Order Traversal II](https://leetcode.com/problems/binary-tree-level-order-traversal-ii/) - Medium
7. [Average of Levels in Binary Tree](https://leetcode.com/problems/average-of-levels-in-binary-tree/) - Easy
8. [Binary Tree Vertical Order Traversal](https://leetcode.com/problems/binary-tree-vertical-order-traversal/) - Medium
9. [Populating Next Right Pointers in Each Node](https://leetcode.com/problems/populating-next-right-pointers-in-each-node/) - Medium
10. [Symmetric Tree](https://leetcode.com/problems/symmetric-tree/) - Easy

## Notes

- Uses queue (FIFO) for processing nodes
- Level-by-level processing is key characteristic
- Useful for minimum path problems in trees

---

**Tags:** #dsa #tree-bfs #leetcode

**Last Reviewed:** 2025-11-05