## 🧠 Stack-based Expression Evaluation

**Stack-based Expression Evaluation** uses stacks to parse and evaluate mathematical expressions, handling operator precedence and parentheses efficiently.

• Converts infix expressions to postfix (RPN) or evaluates directly using operator/operand stacks
• Handles operator precedence, associativity, and parentheses correctly
• Two main approaches: Shunting Yard algorithm and direct evaluation with two stacks
• Time complexity: O(n), Space complexity: O(n) for stack storage

**Related:** [[Stack]] [[Expression Parsing]] [[Mathematics]] [[Parsing]]

---

## 💡 Python Code Snippet

```python
def evaluate_postfix(tokens):
    """
    Evaluate Reverse Polish Notation (postfix) expression
    """
    stack = []
    operators = {'+', '-', '*', '/'}
    
    for token in tokens:
        if token in operators:
            # Pop two operands (order matters for - and /)
            b = stack.pop()
            a = stack.pop()
            
            if token == '+':
                stack.append(a + b)
            elif token == '-':
                stack.append(a - b)
            elif token == '*':
                stack.append(a * b)
            elif token == '/':
                # Truncate towards zero
                stack.append(int(a / b))
        else:
            # Push operand onto stack
            stack.append(int(token))
    
    return stack[0]

def basic_calculator(s):
    """
    Basic calculator supporting +, -, (, ) and spaces
    """
    stack = []
    operand = 0
    result = 0
    sign = 1  # 1 for positive, -1 for negative
    
    for char in s:
        if char.isdigit():
            operand = operand * 10 + int(char)
        elif char == '+':
            result += sign * operand
            sign = 1
            operand = 0
        elif char == '-':
            result += sign * operand
            sign = -1
            operand = 0
        elif char == '(':
            # Push current result and sign onto stack
            stack.append(result)
            stack.append(sign)
            # Reset for new sub-expression
            result = 0
            sign = 1
        elif char == ')':
            # Complete current operand
            result += sign * operand
            # Pop sign and previous result
            result *= stack.pop()  # sign before '('
            result += stack.pop()  # result before '('
            operand = 0
    
    return result + sign * operand

def basic_calculator_ii(s):
    """
    Basic calculator supporting +, -, *, / without parentheses
    """
    stack = []
    operand = 0
    operation = '+'
    
    for i, char in enumerate(s):
        if char.isdigit():
            operand = operand * 10 + int(char)
        
        # Process operation when we hit an operator or reach end
        if char in '+-*/' or i == len(s) - 1:
            if operation == '+':
                stack.append(operand)
            elif operation == '-':
                stack.append(-operand)
            elif operation == '*':
                stack.append(stack.pop() * operand)
            elif operation == '/':
                # Truncate towards zero
                prev = stack.pop()
                stack.append(int(prev / operand))
            
            operation = char
            operand = 0
    
    return sum(stack)

def basic_calculator_iii(s):
    """
    Basic calculator supporting +, -, *, /, (, )
    """
    def calculate(s, index):
        stack = []
        operand = 0
        operation = '+'
        
        while index < len(s):
            char = s[index]
            
            if char.isdigit():
                operand = operand * 10 + int(char)
            elif char == '(':
                # Recursively calculate sub-expression
                operand, index = calculate(s, index + 1)
            elif char in '+-*/)':
                # Process current operand with previous operation
                if operation == '+':
                    stack.append(operand)
                elif operation == '-':
                    stack.append(-operand)
                elif operation == '*':
                    stack.append(stack.pop() * operand)
                elif operation == '/':
                    prev = stack.pop()
                    stack.append(int(prev / operand))
                
                if char == ')':
                    break
                
                operation = char
                operand = 0
            
            index += 1
        
        return sum(stack), index
    
    result, _ = calculate(s, 0)
    return result

def infix_to_postfix(expression):
    """
    Convert infix expression to postfix using Shunting Yard algorithm
    """
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2, '^': 3}
    right_associative = {'^'}
    
    output = []
    operator_stack = []
    
    i = 0
    while i < len(expression):
        char = expression[i]
        
        if char.isalnum():
            # Handle multi-character operands
            operand = ''
            while i < len(expression) and expression[i].isalnum():
                operand += expression[i]
                i += 1
            output.append(operand)
            continue
        elif char == '(':
            operator_stack.append(char)
        elif char == ')':
            # Pop operators until '('
            while operator_stack and operator_stack[-1] != '(':
                output.append(operator_stack.pop())
            operator_stack.pop()  # Remove '('
        elif char in precedence:
            # Pop operators with higher or equal precedence
            while (operator_stack and 
                   operator_stack[-1] != '(' and
                   operator_stack[-1] in precedence and
                   (precedence[operator_stack[-1]] > precedence[char] or
                    (precedence[operator_stack[-1]] == precedence[char] and 
                     char not in right_associative))):
                output.append(operator_stack.pop())
            operator_stack.append(char)
        
        i += 1
    
    # Pop remaining operators
    while operator_stack:
        output.append(operator_stack.pop())
    
    return ' '.join(output)

def evaluate_infix_directly(expression):
    """
    Evaluate infix expression directly using two stacks
    """
    def precedence(op):
        return {'(': 0, '+': 1, '-': 1, '*': 2, '/': 2}[op]
    
    def apply_operator(operators, operands):
        op = operators.pop()
        b = operands.pop()
        a = operands.pop()
        
        if op == '+':
            operands.append(a + b)
        elif op == '-':
            operands.append(a - b)
        elif op == '*':
            operands.append(a * b)
        elif op == '/':
            operands.append(a // b)
    
    operators = []
    operands = []
    i = 0
    
    while i < len(expression):
        char = expression[i]
        
        if char == ' ':
            i += 1
            continue
        elif char.isdigit():
            # Handle multi-digit numbers
            num = 0
            while i < len(expression) and expression[i].isdigit():
                num = num * 10 + int(expression[i])
                i += 1
            operands.append(num)
            continue
        elif char == '(':
            operators.append(char)
        elif char == ')':
            # Apply operators until '('
            while operators[-1] != '(':
                apply_operator(operators, operands)
            operators.pop()  # Remove '('
        else:  # Operator
            # Apply operators with higher or equal precedence
            while (operators and 
                   precedence(operators[-1]) >= precedence(char)):
                apply_operator(operators, operands)
            operators.append(char)
        
        i += 1
    
    # Apply remaining operators
    while operators:
        apply_operator(operators, operands)
    
    return operands[0]

def decode_string(s):
    """
    Decode string like "3[a2[c]]" -> "accaccacc"
    """
    stack = []
    current_string = ''
    current_num = 0
    
    for char in s:
        if char.isdigit():
            current_num = current_num * 10 + int(char)
        elif char == '[':
            # Push current state onto stack
            stack.append((current_string, current_num))
            current_string = ''
            current_num = 0
        elif char == ']':
            # Pop from stack and build string
            prev_string, num = stack.pop()
            current_string = prev_string + current_string * num
        else:
            current_string += char
    
    return current_string

def valid_parentheses(s):
    """
    Check if parentheses are valid and balanced
    """
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    
    for char in s:
        if char in mapping:
            # Closing bracket
            if not stack or stack.pop() != mapping[char]:
                return False
        else:
            # Opening bracket
            stack.append(char)
    
    return len(stack) == 0

def remove_invalid_parentheses(s):
    """
    Remove minimum invalid parentheses to make string valid
    """
    def is_valid(string):
        count = 0
        for char in string:
            if char == '(':
                count += 1
            elif char == ')':
                count -= 1
                if count < 0:
                    return False
        return count == 0
    
    # BFS to find minimum removals
    from collections import deque
    
    queue = deque([s])
    visited = {s}
    found = False
    result = []
    
    while queue and not found:
        for _ in range(len(queue)):
            current = queue.popleft()
            
            if is_valid(current):
                result.append(current)
                found = True
            
            if not found:
                # Generate all possible strings by removing one character
                for i in range(len(current)):
                    if current[i] in '()':
                        next_string = current[:i] + current[i+1:]
                        if next_string not in visited:
                            visited.add(next_string)
                            queue.append(next_string)
    
    return result

def different_ways_add_parentheses(expression):
    """
    Different ways to add parentheses to get different results
    """
    memo = {}
    
    def compute(expression):
        if expression in memo:
            return memo[expression]
        
        result = []
        
        # Try splitting at each operator
        for i, char in enumerate(expression):
            if char in '+-*':
                # Split into left and right parts
                left_results = compute(expression[:i])
                right_results = compute(expression[i+1:])
                
                # Combine results from left and right
                for left in left_results:
                    for right in right_results:
                        if char == '+':
                            result.append(left + right)
                        elif char == '-':
                            result.append(left - right)
                        elif char == '*':
                            result.append(left * right)
        
        # Base case: if no operators, it's just a number
        if not result:
            result.append(int(expression))
        
        memo[expression] = result
        return result
    
    return compute(expression)

def longest_valid_parentheses(s):
    """
    Length of longest valid parentheses substring
    """
    stack = [-1]  # Initialize with -1 to handle edge cases
    max_length = 0
    
    for i, char in enumerate(s):
        if char == '(':
            stack.append(i)
        else:  # char == ')'
            stack.pop()
            
            if not stack:
                # No matching '(' found
                stack.append(i)
            else:
                # Calculate length of current valid parentheses
                length = i - stack[-1]
                max_length = max(max_length, length)
    
    return max_length

def calculator_with_functions(expression):
    """
    Calculator supporting basic operations and functions like max, min
    """
    def parse_function(expr, i):
        # Parse function name
        func_name = ''
        while i < len(expr) and expr[i].isalpha():
            func_name += expr[i]
            i += 1
        
        # Skip '('
        i += 1
        
        # Parse arguments
        args = []
        level = 0
        start = i
        
        while i < len(expr):
            if expr[i] == '(':
                level += 1
            elif expr[i] == ')':
                if level == 0:
                    # End of function
                    if start < i:
                        args.append(evaluate(expr[start:i]))
                    break
                level -= 1
            elif expr[i] == ',' and level == 0:
                # Argument separator
                args.append(evaluate(expr[start:i]))
                start = i + 1
            i += 1
        
        # Apply function
        if func_name == 'max':
            return max(args), i + 1
        elif func_name == 'min':
            return min(args), i + 1
        
        return 0, i + 1
    
    def evaluate(expr):
        # Simple evaluation for this example
        return eval(expr)  # In practice, use proper parsing
    
    # This is a simplified version - full implementation would be more complex
    return evaluate(expression)
```

---

## 🔗 LeetCode Practice Problems (10)

- **Evaluate Reverse Polish Notation** – https://leetcode.com/problems/evaluate-reverse-polish-notation/
- **Basic Calculator** – https://leetcode.com/problems/basic-calculator/
- **Basic Calculator II** – https://leetcode.com/problems/basic-calculator-ii/
- **Basic Calculator III** – https://leetcode.com/problems/basic-calculator-iii/
- **Valid Parentheses** – https://leetcode.com/problems/valid-parentheses/
- **Remove Invalid Parentheses** – https://leetcode.com/problems/remove-invalid-parentheses/
- **Different Ways to Add Parentheses** – https://leetcode.com/problems/different-ways-to-add-parentheses/
- **Longest Valid Parentheses** – https://leetcode.com/problems/longest-valid-parentheses/
- **Decode String** – https://leetcode.com/problems/decode-string/
- **Parse Lisp Expression** – https://leetcode.com/problems/parse-lisp-expression/

---

## 🧠 Flashcard (for spaced repetition)

What is the Stack-based Expression Evaluation pattern? ::

• Uses stacks to parse and evaluate mathematical expressions, handling precedence and parentheses
• Two approaches: convert to postfix (RPN) or direct evaluation with operator/operand stacks
• Applied to calculators, expression parsing, parentheses validation, and mathematical interpreters

[[stack-based-expression-evaluation]] 