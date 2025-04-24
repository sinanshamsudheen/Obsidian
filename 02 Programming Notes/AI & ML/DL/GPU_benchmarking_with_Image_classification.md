
### created: 19-04-2025
---
### Concept  
Explain the core idea in your own words. What is it?
sigmoid function works best in classification problems.
ReLU is prefered for hidden layers.
more hidden = more better (better feature extraction)
when we have one hot encoded output value we use 'categorical crossentropy'
if discrete values, we use 'sparse_categorical_crossentropy'

What is the concept of GPU_benchmarking_with_Image_classification:: basically using GPU for an Image Classification.

---
### Context  
Where and when is it used? Why is it important?

In what context is GPU_benchmarking_with_Image_classification typically applied::basically everywhere we need computational power

---
### Connection  
Link this to related concepts, building blocks, or prerequisites.

- [[Support_Vector_Machine]]

What concepts are connected to GPU_benchmarking_with_Image_classification::[[Topic1]], [[Topic2]]

---
### Concrete Example  
Provide a practical example (code snippet, diagram, equation, or analogy).

```python
# Example Python code
model = keras.Sequential([

keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),

keras.layers.MaxPooling2D((2,2)),

  

keras.layers.Conv2D(64, (3,3), activation='relu'),

keras.layers.MaxPooling2D((2,2)),

  

keras.layers.Flatten(),

keras.layers.Dense(64, activation='relu'),

keras.layers.Dense(10, activation='softmax') # softmax for multi-class

])

model.compile(optimizer='adam',

loss='categorical_crossentropy',

metrics=['accuracy'])

model.fit(X_train_scaled,y_train_categorical, epochs=50)
```

---
### Iterative Thinking
Reflect to deepen your learning.

What’s one thing I’m still unsure about?
the layering, usage of adam, SGD

How can I apply this to a real project or problem?
for image classification projects

What’s a common misconception or edge case?
i thought it's totally unrelated with Machine Learning


The key idea behind GPU_benchmarking_with_Image_classification is {{that GPU is 55 times faster than CPU}}.


---
##### Tags

#ai/GPU_benchmarking_with_Image_classification #ai #python #flashcard
