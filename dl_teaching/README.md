# Transformer Based Models

## Bert

### Structure
- The BERT model uses a learnable embedding layer which contains tree components: the token embedding, the positional embdedding, and the segment embedding.
- The joint embedding of a sequence is passed through multiple encoder blocks (___masked bidrectional multihead attention___).
- The output of the eocoder layers are fed into a feedforward network that consists of two linear layers and a ReLU activation function.
- After the FFN, the outputs are passed through a layer normalization layer and finally softmax for maksed work prediction and next sentence prediction.
<div style="text-align:center">

<img src=https://github.com/anyangml/machine_learning_projects/assets/137014849/3637ecc4-c28e-45f7-89c7-e91de45106f0 width=300 height=190 /><img width=150 height=200 src="https://github.com/anyangml/machine_learning_projects/assets/137014849/e820f6d2-9cef-4319-9b85-6947c9b9ee52">
</div>

### Training
- BERT is an encoder only model, it's trained on two tasks ___Masked language Model___ and ___Next Sentence Prediction___.
- The loss function is the sum of two components: the cross-entropy loss for masked token prediction and the binary classification loss.
- The attention is processed with a mask that labels the tokens need to be predicted. The encoder structure thus has bidirectional information both before and after the maksed token.
- The __NSP__ task is done by running classification on the output embedding of the special token __[CLS]__.
### HuggingFace

<details close>
<summary>References</summary> 
- https://neptune.ai/blog/how-to-code-bert-using-pytorch-tutorial
- https://github.com/ShivamRajSharma/Transformer-Architectures-From-Scratch/blob/master/BERT.py
- https://coaxsoft.com/blog/building-bert-with-pytorch-from-scratch
</details>

#

## GPT

### Structure
- The embedding layer of GPT has two components, the token embedding and the positional embedding. The positional embedding is learnable.
- The attention is not bidirectional as in BERT, a triangle matrix mask (torch.tril) is used to masked out all future tokens for each input sequence chunks to prevent peaking into the future.
- The attention outputs are passed through a layernorm and a FFN and eventually to a softmax function for prediction.

<div style="text-align:center">
<img src=https://github.com/anyangml/machine_learning_projects/assets/137014849/6e45d04c-8738-41f5-8623-f8550ea7fce6 width=300 height=290 />
</div>

### Training
- GPT is a decoder only model.
- GPT is trained in an autoregressive manner, it is trained to predict the next token (___Language Modeling___).
### HuggingFace

#

## Traditional Transformer Model

### Structure
- The original transformer model uses a fix sinusoidal embedding.
- The model is an encoder-decoder model, which means it has both encodel blocks and decoder blocks.
- In the decoder, there is a maksed self-attention layer and a cross encoder-decoder attention layer.

<div style="text-align:center">
<img src=https://github.com/anyangml/machine_learning_projects/assets/137014849/f4734a5a-f1ea-4c0a-b87f-8971da85ad98 width=300 height=390 />
</div>

### Training
- The encoder-decoder transformer is trained on seq-to-seq transformation tasks like English-to-French translation.
- The encoder and decoder will have two different token embeddings for each language vocabulary.
- The encoding is done in parallel through matrix multiplication, while the decoding is done in sequence.
- The decoding starts with a special token, usually __[EOS]__, and the output of self attention layer is used in cross attention with the encoder output. The final decoder output is used as input for the next cycle of self attention.
### HuggingFace

#

# Appendix
## 
<details close>
<summary><b>Activation Functions</b></summary>
| Activation Function         | Formula                                               | Range             | Derivative                                                          | Pros                                                         | Cons                                                                      |
|-----------------------------|-------------------------------------------------------|-------------------|---------------------------------------------------------------------|--------------------------------------------------------------|---------------------------------------------------------------------------|
| Sigmoid                     | $\sigma(x) = \frac{1}{1 + e^{-x}}$                 | (0, 1)            | $\sigma'(x) = \sigma(x) \cdot (1 - \sigma(x))$                 | Smooth, interpretable output; Used in binary classification  | Vanishing gradient problem; Output not zero-centered                        |
| Tanh                        | $\tanh(x) = \frac{2}{1 + e^{-2x}} - 1$             | (-1, 1)           | $\tanh'(x) = 1 - \tanh^2(x)$                                     | Similar to sigmoid, zero-centered; Reduces vanishing gradient | Still susceptible to vanishing gradient problem                             |
| ReLU                        | $ReLU(x) = \max(0, x)$                              | [0, +∞)           | $ReLU'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}$ | Simple, computationally efficient; Addresses vanishing gradient  | Prone to "dying ReLU" problem; Not zero-centered                            |
| Leaky ReLU                  | $LeakyReLU(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha x & \text{if } x \leq 0 \end{cases}$ | (-∞, +∞) | $LeakyReLU'(x) = \begin{cases} 1 & \text{if } x > 0 \\ \alpha & \text{if } x \leq 0 \end{cases}$ | Addresses "dying ReLU" problem; Non-zero slope for negative inputs | Choosing the right slope $\alpha$ is a hyperparameter; Not zero-centered |
| Parametric ReLU             | $PReLU(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha x & \text{if } x \leq 0 \end{cases}$ | (-∞, +∞) | $PReLU'(x) = \begin{cases} 1 & \text{if } x > 0 \\ \alpha & \text{if } x \leq 0 \end{cases}$ | Similar to Leaky ReLU, but $\alpha$ is learned from data       | Requires more computational resources; Introduces more parameters        |
| Exponential Linear Unit (ELU)| $ELU(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha (e^x - 1) & \text{if } x \leq 0 \end{cases}$ | (-$\alpha$, +∞) | $ELU'(x) = \begin{cases} 1 & \text{if } x > 0 \\ ELU(x) + \alpha & \text{if } x \leq 0 \end{cases}$ | Smooth for negative inputs; Reduces vanishing gradient             | Requires more computational resources; Introduces more parameters        |
| Swish                       | $Swish(x) = \frac{x}{1 + e^{-x}}$                  | (0, +∞)           | $Swish'(x) = \frac{e^x (1 + x + e^{-x})}{(1 + e^{-x})^2}$       | Self-gating property; Competitive performance with ReLU variants | Computationally expensive compared to ReLU variants                      |
| GELU                        | $GELU(x) = 0.5x \cdot (1 + \tanh(\sqrt{\frac{2}{\pi}}(x + 0.044715x^3)))$ | (0, +∞)   | $GELU'(x) = 0.5 \cdot \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}\left(x + 0.044715x^3\right)\right)\right) + \frac{0.5x}{1 + e^{-x}} \cdot \left(1 - \tanh^2\left(\sqrt{\frac{2}{\pi}}\left(x + 0.044715x^3\right)\right)\right)$ | Smooth approximation of ReLU; Effective in deep networks        | Computationally expensive; Not zero-centered                               |


</details>


<details close>
<summary><b>Optimizers</b></summary>


| Optimizer          | Description                                            | Key Features                                        | Learning Rate Sensitivity | Popular Use Cases                    |
|--------------------|--------------------------------------------------------|-----------------------------------------------------|---------------------------|--------------------------------------|
| **SGD (Stochastic Gradient Descent)** | Basic optimization algorithm | - Simple to implement<br>- Computationally efficient<br>- Easy to interpret | Sensitive | Linear models, simple neural networks |
| **Momentum**        | Accelerates SGD in relevant directions                | - Accumulates a fraction of past gradients          | Moderate                 | General optimization                 |
| **Nesterov Accelerated Gradient (NAG)** | Improves upon Momentum by considering future gradients | - Similar to Momentum but with lookahead             | Moderate                 | Deep learning, image classification  |
| **Adagrad**         | Adapts learning rates for each parameter individually  | - Adjusts learning rates based on historical gradients | High                    | Sparse data, natural language processing |
| **RMSprop**         | Mitigates Adagrad's rapid learning rate decay         | - Uses a moving average of squared gradients         | Moderate                 | Recurrent Neural Networks (RNNs), LSTMs |
| **Adam**            | Adaptive Moment Estimation                             | - Combines ideas from Momentum and RMSprop            | Moderate                 | Widely used in various applications    |
| **Adadelta**        | Extension of Adagrad and RMSprop                       | - Adapts learning rates based on a moving average of past gradients | Moderate      | Similar use cases as RMSprop          |
| **Nadam**           | Nesterov-accelerated version of Adam                   | - Incorporates Nesterov momentum into Adam           | Moderate                 | Deep learning tasks, optimization     |
| **AdamW**           | Adam with weight decay regularization                  | - Adds weight decay to the Adam optimizer            | Moderate                 | General optimization, deep learning  |

</details>


<details close>
<summary><b>DeepSpeed</b></summary>

  - Data Parallelism: The exact same model is replicated over multiple GPUs, and different batches of data are sent to these GPUs in parallel. The parameters across all GPUs are synchronized.
- Model Parallelism: Large models that cannot fit into one GPU are split into blocks, the same batch of data go through these blocks sequentially. The GPU unitlization efficiency is low.
  
<div style="text-align:center">
<img src=https://github.com/anyangml/machine_learning_projects/assets/137014849/30535c6b-cfdd-49b5-af9a-9c45953840f3 width=3300 height=120 />
</div>

- Pipeline Parallelism: Similar to model parallelism, however, the dataset is further split into microbatches, so that the GPU will have less idel time. When the first model block finishes processing the first microbatch, it starts working on the second microbatch, while the intermediate results of the first microbatch is passed to the second model block.

<div style="text-align:center">
<img src=https://github.com/anyangml/machine_learning_projects/assets/137014849/61878a04-9d1a-45d9-9603-b0fe234e0b07 width=3300 height=120 />
</div>

- ZeRO: Zero redundant optimizer partitions optimizer stats, parameters and gradients arcoss all GPUs, without replicating the model.
</details>


<details close>
<summary><b>Retrival Agumented Generation</b></summary>
reranker
</details>
<details close>
<summary><b>Context Window</b></summary>
- Longformer: sliding window attention...
- Fastattention
</details>