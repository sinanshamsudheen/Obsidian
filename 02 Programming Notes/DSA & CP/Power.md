
### created: 21-04-2025
---
## Problem Overview  
Implement the function **power(b, e)**, which calculates **b** raised to the power of **e** (i.e. be).

**Platform**: GFG 
**Difficulty**: Medium 
**Link**: [Problem URL](https://www.geeksforgeeks.org/batch/gfg-160-problems/track/recursion-and-backtracking-gfg-160/problem/powx-n)

---
### Concept  
What concept(s) does this problem test?  
(e.g. Binary Search, DP, Greedy, Graphs, Trees, etc.)

What is the key concept used in Power::[recursion]

---
### Context  
Where and when is this useful?  
(e.g. Common in interviews, real-world problem class, etc.)

%% In what context does Power apply::[Use-case explanation] %%

---
### Connection  
Which concepts/techniques/other problems are related?

- [[Recursion]]
- [[Backtracking]]


What is Power connected to::[[Linked_Problem_1]], [[Technique]]

---
### Concrete Example  
Explain how the solution works with a sample input/output.

```python
# Example solution
def power(b, e):

    # Base Case: pow(b, 0) = 1
    if e == 0:
        return 1

    if e < 0:
        return 1 / power(b, -e)

    temp = power(b, e // 2)

    if e % 2 == 0:
        return temp * temp
    else:
        return b * temp * temp
        
```

Walk through a sample example of Power::[Step-by-step explanation]


---
## Iterative Thinking

What confused me at first?
[Write it here]

What mistake did I make and how did I fix it?
[Write it here]

What would I ask in a follow-up or variation?
[Write here]

%% The most common mistake in Power is {{[your insight]}}. %%


---
## Time & Space Complexity

Time: O(logn)
Space: O(1)

What is the time and space complexity of Power::Time: O(...), Space: O(...)


---
##### Tags

#dsa/Power #cp #leetcode #gfg #interview #flashcard

