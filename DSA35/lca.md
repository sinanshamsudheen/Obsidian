# Lowest Common Ancestor (LCA)

## Explanation

The Lowest Common Ancestor problem finds the lowest node that has two given nodes as descendants in a tree. This pattern is crucial for tree problems involving relationships between nodes. The LCA can be found efficiently using various techniques including recursion, parent pointers, or binary lifting for multiple queries.

## Algorithm (Word-based)

- For each node, check if it matches either of the target nodes
- Recursively search left and right subtrees
- If both subtrees return non-null values, current node is LCA
- If only one subtree returns value, return that value
- Continue until LCA is found

## Code Template

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def lowest_common_ancestor(root, p, q):
    """
    Find LCA of nodes p and q in binary tree
    """
    # Base case: reached null or found one of the nodes
    if not root or root == p or root == q:
        return root
    
    # Search in left and right subtrees
    left = lowest_common_ancestor(root.left, p, q)
    right = lowest_common_ancestor(root.right, p, q)
    
    # If both left and right return non-null nodes, current node is LCA
    if left and right:
        return root
    
    # Otherwise, return non-null node (or null if both are null)
    return left if left else right

def lca_with_parent_pointers(node1, node2):
    """
    Find LCA when nodes have parent pointers
    """
    # Get paths from nodes to root
    ancestors1 = set()
    current = node1
    
    # Record all ancestors of node1
    while current:
        ancestors1.add(current)
        current = current.parent
    
    # Find first common ancestor
    current = node2
    while current:
        if current in ancestors1:
            return current
        current = current.parent
    
    return None

def lca_bst(root, p, q):
    """
    Find LCA in Binary Search Tree (optimization)
    """
    # Use BST property: p and q are on same side of a node if both < or both >
    if p.val < root.val and q.val < root.val:
        return lca_bst(root.left, p, q)
    elif p.val > root.val and q.val > root.val:
        return lca_bst(root.right, p, q)
    else:
        # If p and q are on different sides, or one equals root, current node is LCA
        return root
```

## Practice Questions

1. [Lowest Common Ancestor of a Binary Tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/) - Medium
2. [Lowest Common Ancestor of a Binary Search Tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/) - Easy
3. [Lowest Common Ancestor of Deepest Leaves](https://leetcode.com/problems/lowest-common-ancestor-of-deepest-leaves/) - Medium
4. [Find Distance in a Binary Tree](https://leetcode.com/problems/find-distance-in-a-binary-tree/) - Medium
5. [Smallest Subtree with all the Deepest Nodes](https://leetcode.com/problems/smallest-subtree-with-all-the-deepest-nodes/) - Medium
6. [All Nodes Distance K in Binary Tree](https://leetcode.com/problems/all-nodes-distance-k-in-binary-tree/) - Medium
7. [Find Elements in a Contaminated Binary Tree](https://leetcode.com/problems/find-elements-in-a-contaminated-binary-tree/) - Medium
8. [Step-By-Step Directions From a Binary Tree Node to Another](https://leetcode.com/problems/step-by-step-directions-from-a-binary-tree-node-to-another/) - Medium
9. [Count Good Nodes in Binary Tree](https://leetcode.com/problems/count-good-nodes-in-binary-tree/) - Medium
10. [Inorder Successor in BST](https://leetcode.com/problems/inorder-successor-in-bst/) - Medium

## Notes

- Recursive approach works for any binary tree
- BST property allows optimization to O(height)
- Parent pointers allow path-based approaches

---

**Tags:** #dsa #lca #tree #leetcode

**Last Reviewed:** 2025-11-05