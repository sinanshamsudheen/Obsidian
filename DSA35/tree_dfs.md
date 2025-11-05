# Tree Depth First Search (DFS)

## Explanation

Tree DFS explores as far as possible along each branch before backtracking. It can be implemented in three ways: preorder (root-left-right), inorder (left-root-right), and postorder (left-right-root). This pattern is effective for tree exploration, path finding, tree validation, and problems requiring exploring all paths from root to leaves.

## Algorithm (Word-based)

- Process current node (preorder), or
- Process left subtree, then current node (inorder), or
- Process left subtree, right subtree, then current node (postorder)
- Recursively apply DFS to left and right subtrees
- Combine results from subtrees as needed

## Code Template

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def dfs_preorder(root):
    """
    Preorder DFS: Root -> Left -> Right
    """
    if not root:
        return []
    
    result = []
    
    def dfs(node):
        if node:
            result.append(node.val)  # Process current
            dfs(node.left)           # Explore left
            dfs(node.right)          # Explore right
    
    dfs(root)
    return result

def dfs_inorder(root):
    """
    Inorder DFS: Left -> Root -> Right
    """
    if not root:
        return []
    
    result = []
    
    def dfs(node):
        if node:
            dfs(node.left)           # Explore left
            result.append(node.val)  # Process current
            dfs(node.right)          # Explore right
    
    dfs(root)
    return result

def dfs_postorder(root):
    """
    Postorder DFS: Left -> Right -> Root
    """
    if not root:
        return []
    
    result = []
    
    def dfs(node):
        if node:
            dfs(node.left)           # Explore left
            dfs(node.right)          # Explore right
            result.append(node.val)  # Process current
    
    dfs(root)
    return result
```

## Practice Questions

1. [Binary Tree Inorder Traversal](https://leetcode.com/problems/binary-tree-inorder-traversal/) - Easy
2. [Binary Tree Preorder Traversal](https://leetcode.com/problems/binary-tree-preorder-traversal/) - Easy
3. [Binary Tree Postorder Traversal](https://leetcode.com/problems/binary-tree-postorder-traversal/) - Easy
4. [Maximum Depth of Binary Tree](https://leetcode.com/problems/maximum-depth-of-binary-tree/) - Easy
5. [Path Sum](https://leetcode.com/problems/path-sum/) - Easy
6. [Path Sum II](https://leetcode.com/problems/path-sum-ii/) - Medium
7. [Sum Root to Leaf Numbers](https://leetcode.com/problems/sum-root-to-leaf-numbers/) - Medium
8. [Binary Tree Maximum Path Sum](https://leetcode.com/problems/binary-tree-maximum-path-sum/) - Hard
9. [Validate Binary Search Tree](https://leetcode.com/problems/validate-binary-search-tree/) - Medium
10. [Lowest Common Ancestor of a Binary Tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/) - Medium

## Notes

- Three main orders: preorder, inorder, postorder
- Recursion is the natural implementation approach
- Can be implemented iteratively with stack

---

**Tags:** #dsa #tree-dfs #leetcode

**Last Reviewed:** 2025-11-05