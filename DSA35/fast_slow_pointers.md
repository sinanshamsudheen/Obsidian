# Fast and Slow Pointers

## Explanation

The fast and slow pointer technique (also known as the tortoise and hare algorithm) uses two pointers moving at different speeds to detect cycles in linked lists or arrays. The fast pointer moves twice as fast as the slow pointer, allowing it to catch up if there's a cycle. This pattern is particularly useful for detecting circular structures.

## Algorithm (Word-based)

- Initialize slow and fast pointers to the start
- Move slow pointer one step at a time
- Move fast pointer two steps at a time
- If pointers meet, there's a cycle
- If fast pointer reaches end, there's no cycle

## Code Template

```python
def fast_slow_pointers(head):
    """
    Detect cycle in linked list using fast and slow pointers
    """
    if not head or not head.next:
        return False
    
    slow = head
    fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        
        if slow == fast:
            return True  # Cycle detected
    
    return False  # No cycle
```

## Practice Questions

1. [Linked List Cycle](https://leetcode.com/problems/linked-list-cycle/) - Easy
2. [Linked List Cycle II](https://leetcode.com/problems/linked-list-cycle-ii/) - Medium
3. [Happy Number](https://leetcode.com/problems/happy-number/) - Easy
4. [Middle of the Linked List](https://leetcode.com/problems/middle-of-the-linked-list/) - Easy
5. [Palindrome Linked List](https://leetcode.com/problems/palindrome-linked-list/) - Easy
6. [Remove Nth Node From End of List](https://leetcode.com/problems/remove-nth-node-from-end-of-list/) - Medium
7. [Find the Duplicate Number](https://leetcode.com/problems/find-the-duplicate-number/) - Medium
8. [Start of Loop in a Linked List](https://leetcode.com/problems/linked-list-cycle-ii/) - Medium
9. [Intersection of Two Linked Lists](https://leetcode.com/problems/intersection-of-two-linked-lists/) - Easy
10. [Reorder List](https://leetcode.com/problems/reorder-list/) - Medium

## Notes

- Also called Floyd's Cycle Detection Algorithm
- Can be used to find the middle element of a linked list
- Useful for palindrome detection in linked lists

---

**Tags:** #dsa #fast-slow-pointers #leetcode

**Last Reviewed:** 2025-11-05