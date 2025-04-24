
### created: 19-04-2025
---
## Problem Overview  
Given the **head** of a singly linked list, your task is to **left rotate** the linked list **k** times.

**Platform**: LeetCode( right rotate ) / GFG ( left rotate )
**Difficulty**:  Medium 
**Link**: https://leetcode.com/problems/rotate-list/

---
### Concept  
What concept(s) does this problem test?  
For Rotate Right problems, just iterate i till (len-k) inside the for loop.
[[Linked Lists]]

What is the key concept used in Rotate_Left_LL:: we iterate till the kth node, connect newhead till the kth node, cut off the next nodes of kth node, connect tail with head. [[Rotate_Left_LL]]

---
### Context  
Where and when is this useful?  
Common in interviews

---
### Connection  
Which concepts/techniques/other problems are related?

- [[Linked Lists]]
- [[Top250]]


What is Rotate_Left_LL connected to::Linked Lists, Top250

---
### Concrete Example  
Explain how the solution works with a sample input/output.

```C++
# Example solution
class Solution {
  public:
    Node* rotateLeft(Node* head, int k) {
        // Your code here
        if(!head || !head->next || k==0)return head;
        
        int len=1;
        Node* tail = head;
        while(tail->next){
            tail=tail->next;
            len++;
        }
        
        k = k % len;
        if(k==0)return head;
        
        //find the kth node
        Node* curr=head;
        for(int i=1;i<k;i++){
            curr=curr->next;
        }
        
        Node* newHead = curr->next;
        curr->next = NULL;
        tail->next = head;
        
        return newHead;
    }
};
```

Walk through a sample example of Rotate_Left_LL::[[Rotate_Left_LL]]


---
## Iterative Thinking

What confused me at first?
how im gonna connect nodes.

What mistake did I make and how did I fix it?
i focused too much on solving taking out the last node and placing it at head for k times.

What would I ask in a follow-up or variation?
do right rotate

The most common mistake in Rotate_Left_LL is {{[your insight]}}.


---
## Time & Space Complexity

Time: O(n)
Space: O(1)

What is the time and space complexity of Rotate_Left_LL::Time: O(n), Space: O(1)


---
##### Tags

#dsa/Rotate_Left_LL #cp #leetcode #gfg #interview #flashcard #Top250

