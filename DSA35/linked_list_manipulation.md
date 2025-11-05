# Linked List Manipulation

## Explanation

Linked list manipulation involves various techniques for traversing, inserting, deleting, and reorganizing nodes in a linked list. These patterns are crucial for problems involving dynamic data structures, pointer manipulation, and scenarios where arrays are less suitable due to insertion/deletion requirements.

## Algorithm (Word-based)

- Use appropriate pointers (slow, fast, previous) based on the problem
- Handle edge cases like empty lists or single nodes
- Update node connections carefully to maintain list structure
- Track necessary information as you traverse
- Return appropriate node (head, modified list, etc.)

## Code Template

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reverse_linked_list(head):
    """
    Reverse a linked list iteratively
    """
    prev = None
    current = head
    
    while current:
        next_temp = current.next  # Store next node
        current.next = prev       # Reverse the link
        prev = current           # Move prev to current
        current = next_temp      # Move to next node
    
    return prev  # New head of reversed list

def detect_cycle(head):
    """
    Detect cycle in linked list using Floyd's algorithm
    """
    if not head or not head.next:
        return False
    
    slow = fast = head
    
    # Fast pointer moves twice as fast as slow
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        
        if slow == fast:
            return True  # Cycle detected
    
    return False  # No cycle

def get_intersection_node(headA, headB):
    """
    Find intersection node of two linked lists
    """
    if not headA or not headB:
        return None
    
    pointer_a = headA
    pointer_b = headB
    
    # When one pointer reaches end, redirect to start of other list
    while pointer_a != pointer_b:
        pointer_a = pointer_a.next if pointer_a else headB
        pointer_b = pointer_b.next if pointer_b else headA
    
    return pointer_a

def remove_nth_from_end(head, n):
    """
    Remove nth node from the end of the list
    """
    # Use two pointers with n distance apart
    dummy = ListNode(0)
    dummy.next = head
    
    first = dummy
    second = dummy
    
    # Advance first pointer n+1 positions
    for _ in range(n + 1):
        first = first.next
    
    # Move both pointers until first reaches the end
    while first:
        first = first.next
        second = second.next
    
    # Remove the nth node from the end
    second.next = second.next.next
    
    return dummy.next

def merge_two_sorted_lists(l1, l2):
    """
    Merge two sorted linked lists
    """
    dummy = ListNode(0)
    current = dummy
    
    while l1 and l2:
        if l1.val <= l2.val:
            current.next = l1
            l1 = l1.next
        else:
            current.next = l2
            l2 = l2.next
        current = current.next
    
    # Attach remaining nodes
    current.next = l1 or l2
    
    return dummy.next

def add_two_numbers(l1, l2):
    """
    Add two numbers represented as linked lists
    """
    dummy = ListNode(0)
    current = dummy
    carry = 0
    
    while l1 or l2 or carry:
        val1 = l1.val if l1 else 0
        val2 = l2.val if l2 else 0
        
        total = val1 + val2 + carry
        carry = total // 10
        current.next = ListNode(total % 10)
        
        current = current.next
        l1 = l1.next if l1 else None
        l2 = l2.next if l2 else None
    
    return dummy.next
```

## Practice Questions

1. [Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/) - Easy
2. [Merge Two Sorted Lists](https://leetcode.com/problems/merge-two-sorted-lists/) - Easy
3. [Linked List Cycle](https://leetcode.com/problems/linked-list-cycle/) - Easy
4. [Remove Nth Node From End of List](https://leetcode.com/problems/remove-nth-node-from-end-of-list/) - Medium
5. [Add Two Numbers](https://leetcode.com/problems/add-two-numbers/) - Medium
6. [Palindrome Linked List](https://leetcode.com/problems/palindrome-linked-list/) - Easy
7. [Intersection of Two Linked Lists](https://leetcode.com/problems/intersection-of-two-linked-lists/) - Easy
8. [Odd Even Linked List](https://leetcode.com/problems/odd-even-linked-list/) - Medium
9. [Reverse Linked List II](https://leetcode.com/problems/reverse-linked-list-ii/) - Medium
10. [Rotate List](https://leetcode.com/problems/rotate-list/) - Medium

## Notes

- Always consider dummy head nodes to simplify edge cases
- Handle null pointers carefully to avoid runtime errors
- Fast-slow pointer technique is useful for finding middle elements

---

**Tags:** #dsa #linked-list #pointers #leetcode

**Last Reviewed:** 2025-11-05