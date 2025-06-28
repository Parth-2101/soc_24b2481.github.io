# soc_24b2481.github.io
# Building ChatGPT from Scratch - Learning Journey

## Week 1: Essential Python Libraries

### NumPy (Numerical Python)
NumPy forms the foundation of scientific computing in Python. It provides support for large, multi-dimensional arrays and matrices, along with mathematical functions to operate on these arrays[29]. NumPy arrays are more efficient than traditional Python lists because they store homogeneous data types, enabling faster computations and reduced memory usage[32]. The core object is the ndarray (n-dimensional array), which allows for vectorized operations and broadcasting capabilities that make numerical computations highly optimized[28].

### Pandas
Pandas is the go-to library for data manipulation and analysis in Python. Built on top of NumPy, it provides two primary data structures: Series (one-dimensional) and DataFrame (two-dimensional)[32]. DataFrames are structured like spreadsheets or database tables, with labeled rows and columns that make data exploration intuitive[29]. Pandas excels at handling missing data, merging datasets, filtering operations, and transforming data structures. It includes many SQL-like operations such as join, merge, group by, and pivot operations, making it indispensable for data preprocessing and analysis[32].

### Matplotlib
Matplotlib is Python's primary plotting library for creating static, animated, and interactive visualizations[29]. It offers extensive customization options for creating publication-quality figures, including line plots, scatter plots, bar charts, histograms, and complex multi-subplot layouts. The library provides both object-oriented and procedural interfaces, allowing users to create everything from simple exploratory plots to sophisticated scientific visualizations[28].

---

## Weeks 2 & 3: Neural Networks Fundamentals

### Introduction to Neural Networks
Neural networks are computational models inspired by biological neural systems that excel at learning complex patterns in data[12]. Unlike linear models that can only fit straight lines, neural networks can approximate curved relationships by combining multiple simple mathematical functions through interconnected nodes (neurons)[7]. Each connection has associated weights and biases that are learned during training, allowing the network to map inputs to outputs through layers of transformations[12].

### Backpropagation
Backpropagation is the fundamental algorithm for training neural networks by optimizing weights and biases[7]. It works by calculating gradients using the chain rule of calculus, starting from the output layer and propagating error signals backward through the network[8]. This process determines how much each parameter contributed to the prediction error and updates them accordingly using gradient descent[7]. The algorithm enables neural networks to learn from their mistakes by iteratively adjusting parameters to minimize the loss function[8].

### Activation Functions
Activation functions introduce non-linearity into neural networks, enabling them to model complex patterns beyond linear relationships[10]. *ReLU (Rectified Linear Unit)* is the most popular activation function, outputting the input directly if positive and zero otherwise, which helps mitigate vanishing gradient problems[13]. *Sigmoid* maps inputs to values between 0 and 1, historically important but less used in deep networks due to saturation issues. *Tanh* provides outputs between -1 and 1, offering zero-centered outputs but still suffering from vanishing gradients in deep architectures[10][13].

### Gradient Descent
Gradient descent is an iterative optimization algorithm that finds parameter values minimizing the loss function[11]. It calculates the derivative (slope) of the loss function with respect to each parameter and takes steps in the direction that reduces the loss[11]. The step size is controlled by the learning rate, determining whether to take small or large updates. The algorithm continues until the step size becomes very small or reaches a maximum number of iterations[11].

### Cross Entropy
Cross entropy is the preferred loss function for classification problems with softmax outputs[9]. Unlike sum of squared residuals used for regression, cross entropy is specifically designed for probability distributions[16]. It heavily penalizes confident wrong predictions while being more forgiving of uncertain predictions, making it ideal for training classification models[19]. The function measures the difference between predicted and true probability distributions[16].

---

## Week 4: Advanced Architectures

### Recurrent Neural Networks (RNNs)
RNNs are specialized architectures for processing sequential data of varying lengths by incorporating feedback loops[21]. Unlike feedforward networks requiring fixed input sizes, RNNs maintain hidden states that carry information from previous time steps[21]. This memory mechanism makes them suitable for tasks involving temporal dependencies, such as language modeling, time series prediction, and sequential pattern recognition[21]. However, basic RNNs suffer from vanishing gradient problems when processing long sequences[22].

### Long Short-Term Memory (LSTM)
LSTMs are advanced RNN variants that solve the vanishing gradient problem through sophisticated gating mechanisms[22]. They maintain both long-term and short-term memory through three key components: *forget gates* determine what percentage of long-term memory to retain, *input gates* decide how much new information to store, and *output gates* control how much memory to output as the current state[22]. These gates use sigmoid activation functions to create percentage-based decisions, allowing LSTMs to selectively remember and forget information over extended sequences[23].

### Word Embeddings and Word2Vec
Word embeddings convert text into numerical representations that capture semantic relationships between words[24]. Rather than using arbitrary numbers, embeddings ensure that similar words have similar vector representations in a high-dimensional space[26]. *Word2Vec* is a neural network approach that learns embeddings by predicting surrounding words in a context window[24]. It uses techniques like Continuous Bag-of-Words (CBOW) and Skip-gram methods, along with negative sampling to efficiently train on large vocabularies by reducing the number of weights optimized per step[24][26].

### Seq2Seq Architecture
Sequence-to-sequence models handle problems where input and output sequences have different lengths, such as machine translation[31]. The architecture consists of an *encoder* that processes the input sequence and creates a context vector summarizing the information, and a *decoder* that uses this context vector to generate the output sequence one token at a time[34]. Both components typically use LSTM or similar recurrent architectures to handle variable-length sequences[31]. However, the fixed-size context vector creates a bottleneck problem for long sequences[31].

### Attention Mechanisms
Attention mechanisms enhance seq2seq models by allowing the decoder to focus on relevant parts of the input sequence rather than relying solely on a fixed context vector[25]. The mechanism computes similarity scores between decoder states and all encoder states, converts these to attention weights using softmax, and creates weighted combinations of encoder states for each decoder step[25]. This approach significantly improves performance on long sequences by providing direct access to all input information and resolving the bottleneck problem[27].

### Transformers
Transformers represent a revolutionary architecture that relies entirely on attention mechanisms, eliminating the need for recurrent connections[30]. Key innovations include *self-attention* that allows each position to attend to all positions in the input sequence, *masked self-attention* that prevents future positions from influencing current predictions during training, *multi-head attention* that applies attention multiple times with different learned projections, and *positional encoding* that provides sequence order information[30][33]. This architecture enables highly parallelizable training and has become the foundation for modern large language models[27]. Decoder-only transformers generate text by predicting one token at a time while attending to all previously generated tokens[30].
