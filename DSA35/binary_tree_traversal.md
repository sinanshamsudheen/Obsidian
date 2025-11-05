# Binary Tree Traversal

## Explanation

Binary tree traversal techniques visit each node in a tree exactly once. The three main recursive traversals are inorder (left-root-right), preorder (root-left-right), and postorder (left-right-root). These patterns form the foundation for solving most binary tree problems and can be implemented recursively or iteratively using stacks.

## Algorithm (Word-based)

- For inorder: Process left subtree, current node, right subtree
- For preorder: Process current node, left subtree, right subtree
- For postorder: Process left subtree, right subtree, current node
- Can be implemented recursively or iteratively with stacks
- Level-order uses queue for breadth-first processing

## Code Template

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# Recursive traversals
def inorder_recursive(root, result=None):
    """
    Inorder traversal: Left -> Root -> Right
    """
    if result is None:
        result = []
    
    if root:
        inorder_recursive(root.left, result)
        result.append(root.val)
        inorder_recursive(root.right, result)
    
    return result

def preorder_recursive(root, result=None):
    """
    Preorder traversal: Root -> Left -> Right
    """
    if result is None:
        result = []
    
    if root:
        result.append(root.val)
        preorder_recursive(root.left, result)
        preorder_recursive(root.right, result)
    
    return result

def postorder_recursive(root, result=None):
    """
    Postorder traversal: Left -> Right -> Root
    """
    if result is None:
        result = []
    
    if root:
        postorder_recursive(root.left, result)
        postorder_recursive(root.right, result)
        result.append(root.val)
    
    return result

# Iterative traversals
def inorder_iterative(root):
    """
    Iterative inorder traversal using stack
    """
    result = []
    stack = []
    current = root
    
    while stack or current:
        # Go to leftmost node
        while current:
            stack.append(current)
            current = current.left
        
        # Process current node
        current = stack.pop()
        result.append(current.val)
        
        # Visit right subtree
        current = current.right
    
    return result

def preorder_iterative(root):
    """
    Iterative preorder traversal using stack
    """
    if not root:
        return []
    
    result = []
    stack = [root]
    
    while stack:
        node = stack.pop()
        result.append(node.val)
        
        # Push right first so left is processed first
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    
    return result
```

## Practice Questions

1. [Binary Tree Inorder Traversal](https://leetcode.com/problems/binary-tree-inorder-traversal/) - Easy
2. [Binary Tree Preorder Traversal](https://leetcode.com/problems/binary-tree-preorder-traversal/) - Easy
3. [Binary Tree Postorder Traversal](https://leetcode.com/problems/binary-tree-postorder-traversal/) - Easy
4. [Binary Tree Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/) - Medium
5. [Binary Tree Zigzag Level Order Traversal](https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/) - Medium
6. [Binary Tree Vertical Order Traversal](https://leetcode.com/problems/binary-tree-vertical-order-traversal/) - Medium
7. [Construct Binary Tree from Preorder and Inorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/) - Medium
8. [Binary Tree Right Side View](https://leetcode.com/problems/binary-tree-right-side-view/) - Medium
9. [Flatten Binary Tree to Linked List](https://leetcode.com/problems/flatten-binary-tree-to-linked-list/) - Medium
10. [Binary Search Tree Iterator](https://leetcode.com/problems/binary-search-tree-iterator/) - Medium

## Notes

- Recursion is natural but can cause stack overflow for deep trees
- Iterative approaches use explicit stacks
- Morris traversal provides O(1) space for inorder (modifies tree)

---

**Tags:** #dsa #binary-tree-traversal #tree #leetcode

**Last Reviewed:** 2025-11-05