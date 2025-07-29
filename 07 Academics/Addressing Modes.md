Here's an updated and complete version of your **Types of Addressing Modes** notes, now including a dedicated **Effective Address (EA)** section for each mode, along with the comparison table at the end.

---

# 📘 Types of Addressing Modes in Computer Architecture

_(With Effective Address explanation)_

---

### 1. ✅ **Immediate Mode**

- **Operand** is given **directly in the instruction**.
    
- 🔍 **EA = Not needed** (no address; operand is directly available).
    
- 💡 _Example:_ `MOV R1, #10` → Copy value 10 directly into R1.
    

---

### 2. ✅ **Register Mode**

- Operand is stored in a **register**.
    
- 🔍 **EA = Register name**
    
- 💡 _Example:_ `MOV R1, R2` → Copy value from R2 to R1.
    

---

### 3. ✅ **Absolute / Direct Mode**

- Instruction gives the **exact memory address** of the operand.
    
- 🔍 **EA = Address field of instruction**
    
- 💡 _Example:_ `MOV R1, 3000` → Access memory at address 3000.
    

---

### 4. ✅ **Indirect Mode**

- Instruction refers to a **memory location or register** that holds the **address of the operand**.
    
- 🔍 **EA = Contents of the register/memory location specified**
    
- 💡 _Example:_ `MOV R1, (R2)` → Use address stored in R2 to get operand.
    

---

### 5. ✅ **Index Mode**

- A **base address** from the instruction is **added to the contents of an index register**.
    
- 🔍 **EA = Base address + contents of index register**
    
- 💡 _Example:_ `MOV R1, A[X]` → Operand is at A + X.
    

---

### 6. ✅ **Base with Index Mode**

- Combines a **base register** and an **index register** to find operand address.
    
- 🔍 **EA = Contents of base register + contents of index register**
    
- 💡 _Example:_ `MOV R1, (R3 + R4)` → Address from R3 + R4.
    

---

### 7. ✅ **Base with Index and Offset Mode**

- Adds an **offset value** to the sum of base and index registers.
    
- 🔍 **EA = Base register + Index register + Offset**
    
- 💡 _Example:_ `MOV R1, (R3 + R4 + 8)` → Address from R3 + R4 + 8.
    

---

### 8. ✅ **Relative Mode**

- Address is given **relative to the current Program Counter (PC)**.
    
- 🔍 **EA = PC + offset (from instruction)**
    
- 💡 _Example:_ `JMP 40` → Jump to PC + 40.
    

---

### 9. ✅ **Auto-Increment Mode**

- Operand is found at the register address, then the register is incremented.
    
- 🔍 **EA = Contents of register; Register = Register + 1 (or word size)**
    
- 💡 _Example:_ `MOV R1, (R2)+` → Use R2, then R2 = R2 + 1
    

---

### 10. ✅ **Auto-Decrement Mode**

- Register is **decremented first**, then the value at that address is used.
    
- 🔍 **EA = Register = Register - 1 (or word size); use EA**
    
- 💡 _Example:_ `MOV R1, -(R2)` → R2 = R2 - 1, then use that address.
    

---

## 📊 Comparison Table of Addressing Modes (with EA)

|Mode|Operand Location|Effective Address (EA)|Memory Accesses|Use Case|Speed|
|---|---|---|---|---|---|
|Immediate|Inside instruction|N/A|0|Constants, immediate ops|Fastest|
|Register|Register|Register name|0|Temporary values|Fast|
|Direct / Absolute|Memory|EA = Address in instruction|1|Access fixed memory|Medium|
|Indirect|Pointer (reg/mem)|EA = [address in reg/mem]|2|Dynamic data, linked lists|Slower|
|Index|Base + index reg|EA = Base + Index|1|Arrays|Medium|
|Base with Index|2 registers|EA = BaseReg + IndexReg|1|Complex access patterns|Medium|
|Base + Index + Offset|2 registers + offset|EA = BaseReg + IndexReg + Offset|1|Structs, object fields|Medium|
|Relative|PC-relative|EA = PC + offset|1|Control flow (branches)|Fast|
|Auto-Increment|Reg then increment|EA = Reg; Reg = Reg + size|1|Iteration (forward)|Fast|
|Auto-Decrement|Decrement then Reg|Reg = Reg - size; EA = Reg|1|Iteration (backward)|Fast|

---


Certainly! Let's go over **Auto-Increment** and **Auto-Decrement** addressing modes with **clear examples** and explain how they work.

---

## 🔄 Auto-Increment Addressing Mode

### 💡 Concept:

- The operand is accessed at the **address stored in a register**.
    
- After the operand is accessed, the register is **automatically incremented** by the size of the operand (usually 1 for byte, 4 for word, etc.).
    

### ✅ Example:

Let's say:

- Register **R1 = 1000**
    
- Memory:  
    `M[1000] = 25`  
    `M[1004] = 30`
    
- Instruction: `LOAD R2, (R1)+` → _(Load value at address in R1 into R2, then increment R1)_
    

### 🔁 Execution:

- `R2 ← M[R1]` → `R2 ← 25`
    
- `R1 ← R1 + 4` → `R1 ← 1004`
    

---

### 📦 Block Diagram (Markdown Style):

```markdown
Before:
R1 → 1000
Memory[1000] = 25

After:
R2 = 25
R1 → 1004
```

### 🟨 Effective Address (EA):

```html
<span style="background-color: yellow; font-weight: bold; padding: 2px 6px; border-radius: 4px;">
EA = [Ri]; Ri ← Ri + d
</span>
```

_(Where `d` = size of operand, like 4 for word)_

---

## 🔁 Auto-Decrement Addressing Mode

### 💡 Concept:

- The register is **first decremented** by the size of the operand.
    
- Then, the operand is accessed at the new address.
    

### ✅ Example:

Let’s say:

- Register **R1 = 1004**
    
- Memory:  
    `M[1000] = 25`  
    `M[1004] = 30`
    
- Instruction: `LOAD R2, -(R1)` → _(Decrement R1, then load from address in R1 into R2)_
    

### 🔁 Execution:

- `R1 ← R1 - 4` → `R1 = 1000`
    
- `R2 ← M[R1]` → `R2 = 25`
    

---

### 📦 Block Diagram (Markdown Style):

```markdown
Before:
R1 → 1004
Memory[1000] = 25

After:
R1 → 1000
R2 = 25
```

### 🟨 Effective Address (EA):

```html
<span style="background-color: yellow; font-weight: bold; padding: 2px 6px; border-radius: 4px;">
Ri ← Ri - d; EA = [Ri]
</span>
```

---

Would you like me to integrate this into the complete addressing modes note as well?
![[Pasted image 20250501143336.png]]
