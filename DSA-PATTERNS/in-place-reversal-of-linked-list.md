## 🧠 In-place Reversal of Linked List

The **In-place Reversal of Linked List** pattern reverses parts or all of a linked list without using extra space.

• Uses three pointers: previous, current, and next to reverse links
• Can reverse entire list or specific portions (between two positions)
• Achieves O(n) time complexity and O(1) space complexity
• Key technique is to reverse the direction of pointers while traversing

**Related:** [[Linked List]]

---

## 💡 Python Code Snippet

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reverse_linked_list(head):
    """
    Reverse entire linked list in-place
    """
    prev = None
    current = head
    
    while current:
        # Store next node before breaking the link
        next_node = current.next
        
        # Reverse the link
        current.next = prev
        
        # Move pointers forward
        prev = current
        current = next_node
    
    return prev  # prev becomes the new head

def reverse_between(head, left, right):
    """
    Reverse linked list between positions left and right (1-indexed)
    Example: 1->2->3->4->5, left=2, right=4 → 1->4->3->2->5
    """
    if not head or left == right:
        return head
    
    # Create dummy node to handle edge cases
    dummy = ListNode(0)
    dummy.next = head
    
    # Find the node before the reversal starts
    prev_start = dummy
    for _ in range(left - 1):
        prev_start = prev_start.next
    
    # Start of reversal section
    start = prev_start.next
    then = start.next
    
    # Reverse the section
    for _ in range(right - left):
        start.next = then.next
        then.next = prev_start.next
        prev_start.next = then
        then = start.next
    
    return dummy.next

def reverse_k_group(head, k):
    """
    Reverse nodes in groups of k
    Example: 1->2->3->4->5, k=3 → 3->2->1->4->5
    """
    # Check if we have at least k nodes
    count = 0
    node = head
    while node and count < k:
        node = node.next
        count += 1
    
    if count == k:
        # Reverse current group
        node = reverse_k_group(node, k)  # Recursively reverse next group
        
        # Reverse current k-group
        while count > 0:
            next_node = head.next
            head.next = node
            node = head
            head = next_node
            count -= 1
        
        head = node
    
    return head

def reverse_alternate_k_nodes(head, k):
    """
    Reverse every alternate k nodes
    """
    if not head or k <= 1:
        return head
    
    current = head
    prev = None
    
    # Reverse first k nodes
    for _ in range(k):
        if not current:
            break
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node
    
    # head now points to the end of first reversed group
    if head:
        head.next = current
        
        # Skip next k nodes
        for _ in range(k):
            if current:
                current = current.next
        
        # Recursively reverse next group
        if current:
            head.next = reverse_alternate_k_nodes(current, k)
    
    return prev
```

---

## 🔗 LeetCode Practice Problems (10)

- **Reverse Linked List** – https://leetcode.com/problems/reverse-linked-list/
- **Reverse Linked List II** – https://leetcode.com/problems/reverse-linked-list-ii/
- **Reverse Nodes in k-Group** – https://leetcode.com/problems/reverse-nodes-in-k-group/
- **Swap Nodes in Pairs** – https://leetcode.com/problems/swap-nodes-in-pairs/
- **Rotate List** – https://leetcode.com/problems/rotate-list/
- **Odd Even Linked List** – https://leetcode.com/problems/odd-even-linked-list/
- **Palindrome Linked List** – https://leetcode.com/problems/palindrome-linked-list/
- **Reorder List** – https://leetcode.com/problems/reorder-list/
- **Partition List** – https://leetcode.com/problems/partition-list/
- **Remove Duplicates from Sorted List II** – https://leetcode.com/problems/remove-duplicates-from-sorted-list-ii/

---

## 🧠 Flashcard (for spaced repetition)

What is the In-place Reversal of Linked List pattern? ::

• Reverses linked list or portions of it using three pointers (prev, current, next) without extra space
• Achieves O(n) time and O(1) space by reversing pointer directions while traversing
• Used for reversing entire lists, specific sections, or groups of nodes in linked lists

[[in-place-reversal-of-linked-list]] 