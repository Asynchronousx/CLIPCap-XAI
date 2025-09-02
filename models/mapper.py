# File in charge of defining the TransformerMapper class.
# This class is used to map the image embeddings to a higher dimensional space, 
# suitable for a model like GPT2. The TransformerMapper is a simple transformer encoder
# that takes an image embedding as input and outputs a higher dimensional embedding.
import torch.nn as nn

# Define the TransformerMapper class
class TransformerMapper(nn.Module):

    # Initialize the class with the input size, output size, hidden size, number of layers,
    # number of heads, and dropout rate.
    def __init__(self, input_size=512, output_size=768, hidden_size=768, num_layers=8, num_heads=8, dropout_rate=0.1):
        
        # Initialize the parent class
        super().__init__()
        
        # Define the input projection layer: this is a linear layer that takes the image embedding
        # as input and outputs a higher dimensional embedding.
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # Define a single transformer encoder layer: this is an encoder layer that takes the
        # input embedding and outputs a higher dimensional embedding suitable for an LLM model. 
        # - We use by default 8 heads because it's a good compromise between the number of heads
        #   and the complexity of the model. Used in the Attention is all you need paper.
        # - The feedforward dimension is 4 times the hidden dimension, which is a good compromise
        #   between the feedforward dimension and the complexity of the model. Used in the Attention
        #   is all you need paper.
        # - The activation function is gelu, which is a good compromise between the activation function
        #   and the complexity of the model. Used in the Attention is all you need paper.
        # - The batch first parameter is set to True because the input is expected to be of shape
        #   (batch_size, sequence_length, embedding_dimension) instead of (sequence_length, batch_size, embedding_dimension).
        #   Used in the Attention is all you need paper and basically matching pytorch's default.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout_rate,
            activation='gelu',
            batch_first=True
        )

        # Define the transformer encoder: this is a composition of the encoder layers.
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Define the output projection layer: this is a linear layer that takes the 
        self.output_proj = nn.Linear(hidden_size, output_size)

        # Define the layer normalization: this is a layer normalization 
        self.layer_norm = nn.LayerNorm(output_size)

    # Define the forward method that will extend the nn.Module.forward method.
    def forward(self, x):

        # If the input is a 2D tensor, we need to add a dimension to the input.
        # This is because the input is expected to be of shape
        # (batch_size, sequence_length, embedding_dimension) instead of (batch_size, embedding_dimension).
        if len(x.shape) == 2:

            # For example, if the input is of shape (batch_size, embedding_dimension),
            # we need to add a dimension to the input to make it of shape
            # (batch_size, 1, embedding_dimension). So the sequence length is 1 (one token = one image!)
            x = x.unsqueeze(1)
            
        # Project the input: We simply forward the input through the arcitecture.
        # First we project the input to the higher dimensional space.   
        x = self.input_proj(x)

        # Then we forward the input through the transformer encoder.
        x = self.transformer(x)

        # Then we project the input to the output space.
        x = self.output_proj(x)

        # Then we normalize the output.
        x = self.layer_norm(x)

        # Return the output.
        return x