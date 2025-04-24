
### created: 20-04-2025
---
## Problem Overview  

You are given a special linked list with **n** nodes where each node has two pointers a **next pointer** that points to the next node of the singly linked list, and a **random pointer** that points to the random node of the linked list.  

Construct a **copy** of this linked list. The copy should consist of the same number of new nodes, where each new node has the value corresponding to its original node. Both the next and random pointer of the new nodes should point to new nodes in the copied list, such that it also represent the same list state. None of the pointers in the new list should point to nodes in the original list.

Return the **head** of the copied linked list.

**NOTE :** Original linked list should remain unchanged.

**Platform**: GFG 
**Difficulty**:  Hard  
**Link**: [Problem URL](https://www.geeksforgeeks.org/batch/gfg-160-problems/track/linked-list-gfg-160/problem/clone-a-linked-list-with-next-and-random-pointer)

---
### Concept  
What concept(s) does this problem test?  

What is the key concept used in Clone_LL ?:: the idea is to store the newly made nodes inside a hash map as key = real node, value=new node pairs. then in the next loop, connect the next and random of nodes within the map based on the connections of the list. [[Clone_LL]]

---
### Context  
Where and when is this useful?  
(e.g. Common in interviews, real-world problem class, etc.)

%% In what context does Clone_LL apply::[Use-case explanation] %%

---
### Connection  
Which concepts/techniques/other problems are related?

- [[Linked Lists]]
- [[Hash map]]

%% What is Clone_LL connected to::Linked Lists, Hashmaps %%

---
### Concrete Example  
Explain how the solution works with a sample input/output.

```C++
# Example solution
class Solution {
  public:
    Node *cloneLinkedList(Node *head) {
        // Write your code here
        unordered_map<Node*, Node*> mp;
        Node* curr=head;
        
        while(curr){
            mp[curr]=new Node(curr->data);
            curr=curr->next;
        }
        curr=head;
        
        while(curr){
            mp[curr]->next = mp[curr->next];
            mp[curr]->random = mp[curr->random];
            curr=curr->next;
        }
        return mp[head];
    }
};
```

Walk through a sample example of Clone_LL::[Step-by-step explanation]


---
## Iterative Thinking

What confused me at first?
how im gonna copy a list without making a copy of it.

What mistake did I make and how did I fix it?
Tried to solve it in my own way which caused copies of original list(which aint allowed)

What would I ask in a follow-up or variation?
why didnt the cloning and inserting at front method work?

The most common mistake in Clone_LL is {{[not using hash maps]}}.


---
## Time & Space Complexity

Time: O(n)
Space: O(n)

What is the time and space complexity of Clone_LL::Time: O(n), Space: O(n)


---
##### Tags

#dsa/Clone_LL #cp #leetcode #gfg #interview #flashcard

