Sure! Here's a clean and exam-ready explanation of **Optimization using Gradient Descent**:

---

### Optimization using Gradient Descent

**Optimization** in Machine Learning refers to the process of adjusting parameters to minimize (or maximize) a function, typically a **loss** or **cost function** that measures how well a model performs.

**Gradient Descent** is one of the most popular optimization algorithms. It is an iterative method used to find the minimum of a function.

#### Key Idea:

- The **gradient** (or derivative) of a function points in the direction of the **steepest increase**.
- To **minimize** the function, we move in the **opposite direction** of the gradient.

#### Steps of Gradient Descent:

1. **Initialize** the parameters (weights) randomly.
2. **Compute the gradient** of the loss function with respect to each parameter.
3. **Update the parameters** by moving them slightly in the opposite direction of the gradient.

```latex

   \theta = \theta - \eta \nabla L(\theta)
```

- = parameters (weights)
- = learning rate (controls step size)
- = gradient of loss function with respect to

4. **Repeat** steps 2 and 3 until convergence (i.e., until changes become very small or a set number of iterations is reached).

#### Types of Gradient Descent:

- **Batch Gradient Descent**: Uses the whole dataset to compute the gradient.
- **Stochastic Gradient Descent (SGD)**: Uses one data point at a time.
- **Mini-batch Gradient Descent**: Uses a small batch of data points.

#### Intuition:

Imagine you are on a mountain and want to reach the lowest point (valley). You feel the slope (gradient) at your feet and take a step downhill (opposite to the slope). Repeating this process gradually takes you to the bottom.

#### Important Points:

- A **small learning rate** makes convergence slow but steady.
- A **large learning rate** may overshoot the minimum or cause divergence.
- The loss surface may have **local minima**, **saddle points**, and **plateaus** which can affect optimization.

---

Would you also like a quick **summary** version (like 5-6 bullet points) you can revise right before the exam? It often helps!