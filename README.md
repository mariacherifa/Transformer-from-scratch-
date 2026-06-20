# Transformer-from-scratch-
A step-by-step implementation of a Transformer language model from scratch for next-token prediction using PyTorch.

The goal of this project is not only to build a working Transformer, but also to understand how each component is implemented and how they interact together to perform language modeling.

# Vocabulary and tokenization step 
The first step is to transform raw text into a format that can be processed by a neural network.

We begin by creating a vocabulary, which is the set of all unique characters appearing in the text. Each character is assigned a unique integer identifier through a string-to-index dictionary (stoi) and an index-to-string dictionary (itos).

We then implement:
 - an **encoding** function that converts text into a sequence of integer token IDs,
 - a **decoding** function that converts token IDs back into text.

At this stage, the model does not see characters directly; it only sees sequences of integers.

# Data creation and Batching
Once the text has been tokenized, we convert it into a PyTorch tensor. The dataset is then split into: a training set, a validation set.

To train a language model, we need input-target pairs. For a sequence "hello", the model receives

$$\text{hell}$$

as input and learns to predict

$$\text{ello}$$

as targets.

To do this efficiently, we create mini-batches.

Each batch contains:

 - batch_size sequences,
 - each sequence containing block_size tokens.

The parameter block_size represents the maximum context length available to the model.

# Token and Positional Embeddings
Neural networks cannot directly process integer token IDs. Therefore, each token is mapped to a learned vector representation using an embedding table. If

$$\text{idx.shape } = (B, T)$$

then token embeddings have shape

$$(B, T, n_{\text{embd}})$$

where $n_{\text{embd}}$ is the embedding dimension. Since attention alone does not contain any notion of order, we also learn positional embeddings.

These embeddings encode the position of each token inside the sequence and are added to the token embeddings. The resulting representation contains both:

 - the identity of the token,
 - its position within the context.

# Single Attention Head
This is the core building block of the Transformer.

For each token, we construct:

 - a query vector,
 - a key vector,
 - a value vector.

The attention mechanism computes similarity scores between queries and keys:

$$QK^\top$$

which determine how much attention one token should pay to another.

The scores are normalized with a softmax function to obtain attention weights.

Finally, each token builds a new representation as a weighted average of the value vectors.

An important subtlety is the use of a causal mask. During training, a token is only allowed to attend to itself and previous tokens. Future tokens are masked to preserve the autoregressive nature of language modeling.

# Multi-Head Attention
Natural language contains many different types of relationships. For example, one token may need to attend to:

 - nearby words,
 - grammatical dependencies,
 - long-range semantic information.

A single attention head is often insufficient to capture all these patterns. To address this, we create several attention heads in parallel. Each head learns its own attention mechanism and produces its own representation.

The outputs of all heads are:

 - concatenated,
 - projected back to the embedding dimension.

This projection allows information from different heads to be mixed together while keeping a fixed representation size.

# Feed-Forward Network
After attention, each token has gathered information from the surrounding context. The next step is to process this information independently for each token. This is done using a small multilayer perceptron (MLP):

$$n_{\text{embd}} → 4 n_{\text{embd}} → n_{\text{embd}}$$

Unlike attention, this operation does not mix information between tokens. Each token is transformed independently using the same feed-forward network.

# Transformer Block 
A Transformer block combines:

 - multi-head attention,
 - feed-forward network,
 - residual connections,
 - layer normalization.

The structure is:

LayerNorm
 - → Multi-Head Attention
 - → Residual Connection

LayerNorm
 - → Feed Forward Network
 - → Residual Connection

Residual connections help stabilize optimization and allow information to flow through deep networks. Layer normalization improves training stability by keeping activations at a reasonable scale. Multiple Transformer blocks are stacked together to progressively build richer contextual representations.

# Language Modeling Head
After passing through all Transformer blocks, each token is represented by **a contextual embedding**. To predict the next token, we apply a final linear layer that maps each token representation to a vector of vocabulary-sized logits. If the vocabulary size is $V$, then each token produces:

$$(V,)$$

logits corresponding to the scores of all possible next tokens. These logits are transformed into probabilities using a softmax function.

# Training 
The model is trained using the cross-entropy loss. For each token position, the predicted probability distribution is compared to the true next token. The loss is then backpropagated through the network, and the parameters are updated using the AdamW optimizer.

Training proceeds through the standard loop:

 - sample a batch,
 - compute predictions,
 - compute loss,
 - backpropagate gradients,
 - update parameters.

# Evaluation 
To monitor learning, we periodically evaluate the model on both:

 - the training set,
 - the validation set.

The validation loss helps detect overfitting and provides a better estimate of how well the model generalizes to unseen text.

# Text Generation 
Once training is complete, the model can generate text autoregressively.

Starting from an initial context, the model:
 - predicts the probability distribution of the next token,
 - samples a token from this distribution,
 - appends the token to the context,
 - repeats the process.

This iterative procedure allows the Transformer to generate arbitrary-length text one token at a time.
