![[Pasted image 20250501151715.png]]

The figure depicts the internal organization of a processor using a **single-bus architecture**. Here's a breakdown of the key components and their interactions:

---

### **1. Single Internal Bus**
- The **internal processor bus** (highlighted as a central horizontal line) connects all major components. This shared bus allows data transfer between registers, the ALU, and memory interfaces but permits only one transfer at a time, creating a potential bottleneck.

---

### **2. Registers**
- **General-Purpose Registers (R0 to R(n-1))**: Store intermediate data or operands for computations.  
- **Special Registers**:  
  - **PC (Program Counter)**: Holds the address of the next instruction to fetch.  
  - **IR (Instruction Register)**: Stores the currently executing instruction.  
  - **TEMP**: Temporary storage for ALU operations or data transfers.  
  - **MAR (Memory Address Register)**: Holds memory addresses for reads/writes.  
  - **MDR (Memory Data Register)**: Buffers data during memory transfers.  

---

### **3. ALU (Arithmetic Logic Unit)**
- Performs operations (Add, Subtract, XOR, etc.) on data from registers (A and B inputs).  
- **Control Lines**: Determine the operation (e.g., Add/Sub) based on the instruction.  
- **Constant 4**: Often used to increment the PC (e.g., for sequential instruction fetching).  

---

### **4. Memory Interface**
- **Memory Bus**: Connects the processor to external memory.  
- **MAR/MDR**: Bridge between internal bus and memory:  
  - **MAR** sends addresses to memory via *address lines*.  
  - **MDR** sends/receives data via *data lines*.  

---

### **5. Control Unit**
- **Instruction Decoder & Control Logic**: Decodes instructions (from IR) and generates **control signals** to coordinate data movement (e.g., register selection, ALU operations, memory access).  

---

### **6. Data Flow Example**
1. **Fetch**: PC → MAR → Memory → MDR → IR (via internal bus).  
2. **Execute**: Operands from registers (e.g., R0, R1) → ALU → Result stored back (e.g., R2).  
3. **Memory Access**: MAR sends address; data is read into MDR → register.  

---

### **Key Takeaways**
- **Single-Bus Limitation**: Only one data transfer can occur at a time, impacting performance.  
- **Control Dependency**: The Control Unit orchestrates all operations via signals.  
- **Modular Design**: Components like ALU, registers, and memory interfaces are interconnected but functionally distinct.  

This organization is foundational for understanding how processors execute instructions sequentially, balancing computation, data movement, and control.

