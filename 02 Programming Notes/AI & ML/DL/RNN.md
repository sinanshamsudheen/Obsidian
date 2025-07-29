Recurrent Neural Networks
CNN are mainly for Images
RNN are mainly for NLP

Applications: Gmail Auto complete on mails, Google translate, Named Entity Recognition

Why not just use (ANN) nueral Networks for predictions?
1- No fixed size of neurons in a layer
2- too much computation(vectorization)
3- Parameters are not shared

# Named Entity Recognition
Inorder to recognize named entities.(eg: a person)
![[Pasted image 20250604141211.png]]
![[Pasted image 20250604141252.png]]
(same layer at different stages)


![[Pasted image 20250604141503.png]]

# Language Translation
![[Pasted image 20250604141748.png]]

# Deep RNN
![[Pasted image 20250604141818.png]] 
(Multiple hidden layers)

# Types of RNN
Many to Many - Many inputs and outputs (dhaval loves baby yoda)

Many to One - Many inputs to one output (Sentiment analysis: input - paragraph, output - product review)

One to Many - generate songs from a single note.

# Bidirectional RNN
![[Pasted image 20250604163441.png]]