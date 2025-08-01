## 🧠 Tree DFS

**Tree DFS (Depth-First Search)** traverses a tree by going as deep as possible before backtracking, using recursion or a stack.

• Three main types: Pre-order (root → left → right), In-order (left → root → right), Post-order (left → right → root)
• Uses recursion (implicit stack) or explicit stack for iterative approach
• Useful for tree validation, path finding, and tree reconstruction problems
• Time complexity: O(n), Space complexity: O(h) where h is tree height

**Related:** [[Tree]] [[Stack]] [[DFS]] [[Recursion]]

---

## 💡 Python Code Snippet

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def preorder_traversal(root):
    """
    Pre-order: Root → Left → Right
    Useful for creating copy of tree or prefix expressions
    """
    result = []
    
    def dfs(node):
        if not node:
            return
        
        result.append(node.val)  # Process root
        dfs(node.left)           # Traverse left
        dfs(node.right)          # Traverse right
    
    dfs(root)
    return result

def inorder_traversal(root):
    """
    In-order: Left → Root → Right
    For BST, gives sorted order of values
    """
    result = []
    
    def dfs(node):
        if not node:
            return
        
        dfs(node.left)           # Traverse left
        result.append(node.val)  # Process root
        dfs(node.right)          # Traverse right
    
    dfs(root)
    return result

def postorder_traversal(root):
    """
    Post-order: Left → Right → Root
    Useful for deleting tree or getting size
    """
    result = []
    
    def dfs(node):
        if not node:
            return
        
        dfs(node.left)           # Traverse left
        dfs(node.right)          # Traverse right
        result.append(node.val)  # Process root
    
    dfs(root)
    return result

def has_path_sum(root, target_sum):
    """
    Check if tree has root-to-leaf path with given sum
    """
    if not root:
        return False
    
    # If leaf node, check if remaining sum equals node value
    if not root.left and not root.right:
        return target_sum == root.val
    
    # Recursively check left and right subtrees
    remaining_sum = target_sum - root.val
    return (has_path_sum(root.left, remaining_sum) or 
            has_path_sum(root.right, remaining_sum))

def find_all_paths(root, target_sum):
    """
    Find all root-to-leaf paths with given sum
    """
    all_paths = []
    
    def find_paths(node, target, current_path):
        if not node:
            return
        
        # Add current node to path
        current_path.append(node.val)
        
        # If leaf node and sum matches, add path to result
        if not node.left and not node.right and target == node.val:
            all_paths.append(list(current_path))
        else:
            # Recursively search left and right subtrees
            find_paths(node.left, target - node.val, current_path)
            find_paths(node.right, target - node.val, current_path)
        
        # Backtrack: remove current node from path
        current_path.pop()
    
    find_paths(root, target_sum, [])
    return all_paths

def max_path_sum(root):
    """
    Find maximum path sum in binary tree (path can start and end at any nodes)
    """
    max_sum = float('-inf')
    
    def max_gain(node):
        nonlocal max_sum
        
        if not node:
            return 0
        
        # Maximum gain from left and right subtrees (ignore negative gains)
        left_gain = max(max_gain(node.left), 0)
        right_gain = max(max_gain(node.right), 0)
        
        # Price of new path through current node
        price_new_path = node.val + left_gain + right_gain
        
        # Update global maximum
        max_sum = max(max_sum, price_new_path)
        
        # Return maximum gain if we continue path through this node
        return node.val + max(left_gain, right_gain)
    
    max_gain(root)
    return max_sum
```

---

## 🔗 LeetCode Practice Problems (10)

- **Binary Tree Inorder Traversal** – https://leetcode.com/problems/binary-tree-inorder-traversal/
- **Binary Tree Preorder Traversal** – https://leetcode.com/problems/binary-tree-preorder-traversal/
- **Binary Tree Postorder Traversal** – https://leetcode.com/problems/binary-tree-postorder-traversal/
- **Path Sum** – https://leetcode.com/problems/path-sum/
- **Path Sum II** – https://leetcode.com/problems/path-sum-ii/
- **Binary Tree Maximum Path Sum** – https://leetcode.com/problems/binary-tree-maximum-path-sum/
- **Diameter of Binary Tree** – https://leetcode.com/problems/diameter-of-binary-tree/
- **Validate Binary Search Tree** – https://leetcode.com/problems/validate-binary-search-tree/
- **Sum Root to Leaf Numbers** – https://leetcode.com/problems/sum-root-to-leaf-numbers/
- **Count Good Nodes in Binary Tree** – https://leetcode.com/problems/count-good-nodes-in-binary-tree/

---

## 🧠 Flashcard (for spaced repetition)

What is the Tree DFS pattern? ::

• Traverses tree depth-first using recursion or stack with three orders: pre-order, in-order, post-order
• Time O(n), Space O(h) where h is height; uses implicit/explicit stack
• Used for tree validation, path problems, and tree reconstruction tasks

[[tree-dfs]] 