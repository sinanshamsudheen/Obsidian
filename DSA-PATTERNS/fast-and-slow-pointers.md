## 🧠 Fast & Slow Pointers

The **Fast & Slow Pointers** pattern (also called Floyd's Cycle Detection) uses two pointers moving at different speeds through a data structure.

• Fast pointer moves 2 steps, slow pointer moves 1 step at a time
• Primarily used for detecting cycles in linked lists
• Can find the middle of a linked list, detect loops, and find loop start points
• Works because if there's a cycle, fast pointer will eventually catch up to slow pointer

**Related:** [[Linked List]]

---

## 💡 Python Code Snippet

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def has_cycle(head):
    """
    Detect if linked list has a cycle using Floyd's algorithm
    """
    if not head or not head.next:
        return False
    
    slow = head
    fast = head
    
    # Move slow by 1, fast by 2
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        
        # If they meet, there's a cycle
        if slow == fast:
            return True
    
    return False

def find_middle(head):
    """
    Find middle node of linked list
    When fast reaches end, slow will be at middle
    """
    if not head:
        return None
    
    slow = head
    fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    
    return slow

def detect_cycle_start(head):
    """
    Find the starting node of cycle in linked list
    """
    if not head or not head.next:
        return None
    
    # Phase 1: Detect cycle
    slow = head
    fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            break
    else:
        return None  # No cycle
    
    # Phase 2: Find cycle start
    # Move one pointer to head, keep other at meeting point
    # Move both at same speed until they meet
    slow = head
    while slow != fast:
        slow = slow.next
        fast = fast.next
    
    return slow  # This is the cycle start
```

---

## 🔗 LeetCode Practice Problems (10)

- **Linked List Cycle** – https://leetcode.com/problems/linked-list-cycle/
- **Linked List Cycle II** – https://leetcode.com/problems/linked-list-cycle-ii/
- **Find the Duplicate Number** – https://leetcode.com/problems/find-the-duplicate-number/
- **Happy Number** – https://leetcode.com/problems/happy-number/
- **Middle of the Linked List** – https://leetcode.com/problems/middle-of-the-linked-list/
- **Remove Nth Node From End of List** – https://leetcode.com/problems/remove-nth-node-from-end-of-list/
- **Palindrome Linked List** – https://leetcode.com/problems/palindrome-linked-list/
- **Intersection of Two Linked Lists** – https://leetcode.com/problems/intersection-of-two-linked-lists/
- **Start of LinkedList Cycle** – https://leetcode.com/problems/linked-list-cycle-ii/
- **Circular Array Loop** – https://leetcode.com/problems/circular-array-loop/

---

## 🧠 Flashcard (for spaced repetition)

What is the Fast & Slow Pointers pattern? :: • Uses two pointers moving at different speeds (fast: 2 steps, slow: 1 step) through data structure • Primarily for cycle detection in linked lists using Floyd's algorithm • When cycle exists, fast pointer eventually catches up to slow pointer [[fast-and-slow-pointers]] 