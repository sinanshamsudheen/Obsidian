# In-place Reversal of a LinkedList

## Explanation

The in-place reversal of a linked list pattern involves reversing nodes without using additional space. This technique is used for reversing parts of a linked list or the entire list by adjusting pointers. It typically uses three pointers: previous, current, and next to maintain references during the reversal process.

## Algorithm (Word-based)

- Initialize previous pointer as null
- Initialize current pointer at head
- Iterate through the list
- Store next node before changing current's next pointer
- Reverse the link by pointing current to previous
- Move previous and current pointers forward
- Update head to point to new first node

## Code Template

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reverse_linked_list(head):
    """
    Reverse a linked list in-place
    """
    prev = None
    current = head
    
    while current:
        next_temp = current.next  # Store next node
        current.next = prev       # Reverse the link
        prev = current           # Move prev to current
        current = next_temp      # Move to next node
    
    return prev  # New head of reversed list
```

## Practice Questions

1. [Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/) - Easy
2. [Reverse Linked List II](https://leetcode.com/problems/reverse-linked-list-ii/) - Medium
3. [Palindrome Linked List](https://leetcode.com/problems/palindrome-linked-list/) - Easy
4. [Rotate List](https://leetcode.com/problems/rotate-list/) - Medium
5. [Swap Nodes in Pairs](https://leetcode.com/problems/swap-nodes-in-pairs/) - Medium
6. [Odd Even Linked List](https://leetcode.com/problems/odd-even-linked-list/) - Medium
7. [Reverse Nodes in k-Group](https://leetcode.com/problems/reverse-nodes-in-k-group/) - Hard
8. [Remove Linked List Elements](https://leetcode.com/problems/remove-linked-list-elements/) - Easy
9. [Remove Duplicates from Sorted List II](https://leetcode.com/problems/remove-duplicates-from-sorted-list-ii/) - Medium
10. [Reorder List](https://leetcode.com/problems/reorder-list/) - Medium

## Notes

- Always keep track of the next node before changing pointers
- The reversed list starts with the 'prev' pointer
- Use dummy nodes when reversing partial lists

---

**Tags:** #dsa #linkedlist-reversal #leetcode

**Last Reviewed:** 2025-11-05