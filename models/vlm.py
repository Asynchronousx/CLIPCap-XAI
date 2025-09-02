import torch
import logging
import torch.nn as nn

from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from utils.data import Dataset
from models.clip import CLIPW
from utils.knowledge import captions
from torch.utils.data import DataLoader
from models.mapper import TransformerMapper
from transformers import OPTForCausalLM, AutoTokenizer

# Class for our Vision Language Model with OPT-125M as the language model and CLIP as the visual reasoning model.
class ClipCaptioner(nn.Module):
    def __init__(self, prefix_length=10, clip_length=512, hidden_size=768, num_layers=8, num_heads=8, max_length=77):
        super().__init__()
        
        # Determine the device internally 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Prefix length is the number of visual tokens that we want to use as prefix.
        self.prefix_length = prefix_length

        # Max length is the maximum length of the caption. 
        self.max_length = max_length
        
        # Initialize OPT-125M model and tokenizer
        self.opt = OPTForCausalLM.from_pretrained("facebook/opt-125m").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        
        # The tokenizer needs a pad token, if it's not set, we set it to the eos token.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Get embedding size from the language model: the embedding size it's given by the embedding layer of the model.
        # The embedding layer is the layer that maps the tokens to the embeddings.
        self.opt_embedding_size = self.opt.get_input_embeddings().weight.shape[1]

        # Logger initializing for displaying infos
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Initialize our projection mapper and put in on the device of choice.
        # The projection mapper is the layer that maps the prefix to the same embedding size as the tokens.
        # The output size it's just the embedding size of the tokens times the prefix length (10, 768) that will 
        self.clip_project = TransformerMapper(
            input_size=clip_length,
            output_size=self.opt_embedding_size * prefix_length,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads
        ).to(self.device)

        # Create a CLIPW instance to perform image and text analysis while predicting
        self.clip_wrapper = CLIPW(captions=captions["fruits"], device=self.device)

        
        # REMOVED: self.to(self.device).float()
        # Let each component manage its precision. If OPT/Mapper need float32,
        # ensure they are loaded/cast appropriately above or within their classes.
        # Example: self.opt = self.opt.float() if needed.
    
    def forward(self, tokens, prefix, mask=None, labels=None):

        # Tokens are the captions of the images, while prefix is the CLIP embedding of the image.
        # We embed the tokens and the prefix. 
        # The embedding_text is the embedding of the tokens, done with the OPT model.
        # Note that since OPT is a decoder model, the embedding_text is the embedding of the tokens, 
        # done with the OPT model itself.
        embedding_text = self.opt.model.decoder.embed_tokens(tokens)
        
        # Batch size is the number of images in the batch.
        batch_size = prefix.size(0)
        
        # Ensure prefix is float32 to match the clip_project layer's expected dtype
        prefix = prefix.float() 

        # We now need to project the prefix to the same embedding size as the tokens.
        # This is done with the clip_project instance, our projection mapper. 
        prefix_projections = self.clip_project(prefix)

        # We now need to reshape the prefix projections to the same shape as the embedding_text.
        # We use the view function to reshape the prefix projections to the same shape as the embedding_text.
        # Without changing the data.
        prefix_projections = prefix_projections.view(
            batch_size, 
            self.prefix_length, 
            self.opt_embedding_size
        )
        
        # Concatenate prefix projections and embedding text.
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        
        # Adjust attention mask to account for prefix: This happens only if the mask is not None, 
        # (which is the case when we're training the model). Otherwise, in the case of inference, 
        # the mask is not needed.
        if mask is not None:

            # Create prefix mask (all ones): that means that the prefix is not masked.
            # This is done to ensure that the prefix is considered in the attention mechanism
            # since we are actually training the model to predict the prefix.
            prefix_mask = torch.ones(
                batch_size, self.prefix_length,
                dtype=mask.dtype,
                device=mask.device
            )

            # Concatenate with existing mask: we need to match the full sequence length of the tokens: 
            # embeddings + caption. If we did not do this, the model would not know what to do with the prefix.
            mask = torch.cat((prefix_mask, mask), dim=1)
        
        # The labels are the tokens of the captions, so they are the same as the tokens.
        # As above, this happens only if the labels are not None, (which is the case when we're training the model).
        # Otherwise, in the case of inference, the labels are not needed since we're actually trying to generate them.
        if labels is not None:  

            # Create a dummy token (all zeros): 
            # Since we're using a prefix, we need to create a dummy token to match the sequence length of the tokens.
            # Basically we need to create a token that is irrelevant to the model, since we don't want to predict the prefix.
            # We're interested in predicting only the caption.
            dummy_token = torch.zeros(
                tokens.shape[0], self.prefix_length, 
                dtype=torch.int64, device=tokens.device
            )

            # Concatenate the dummy token with the tokens to match the sequence length of the tokens.
            labels = torch.cat((dummy_token, tokens), dim=1)
        
        # Forward pass through the OPT model: We pass the new concatenated embeddings, labels and mask.
        out = self.opt(
            inputs_embeds=embedding_cat,
            labels=labels,
            attention_mask=mask,
            return_dict=True
        )
        
        # Return the output of the OPT model.
        return out
    
    # Function to preprocess the dataset using the CLIP wrapper.
    def preprocess_dataset(self, root_dir, output_dir, batch_size=32):

        # Preprocess the dataset using the CLIP wrapper.
        self.clip_wrapper.preprocess_dataset(root_dir, output_dir, batch_size)

    # Function to only retrieve the visual explanation of the image.
    # For a more, detailed explanation fo the parameters refer to the CLIPW.visualize function.
    def get_visual_explanation(self, image, k=5, plot_individual=True, combined_threshold=0.4, combined_alpha=1.0, normalize_alpha_scaling=False):
        return self.clip_wrapper.visualize(image=image, 
                                           k=k, plot_individual=plot_individual, 
                                           combined_threshold=combined_threshold, 
                                           combined_alpha=combined_alpha, 
                                           normalize_alpha_scaling=normalize_alpha_scaling)

    # Function that generate a caption given an image path using manual token-by-token generation.
    # Uses sampling with temperature, top-k, and top-p filtering and ncludes retry logic for invalid captions.
    def generate(self, 
                 image, 
                 max_length=77, 
                 temperature=0.7, 
                 top_k=50, 
                 top_p=0.9, 
                 prefix_text="A plant disease description: ", 
                 min_length=10,
                 k=5,
                 plot_individual=True,
                 combined_threshold=0.4,
                 combined_alpha=0.7,
                 normalize_alpha_scaling=False):
        
        # Brief description of the arguments of the function since they are not self-explanatory.
        """ 
        Args:
            image (str, other): The path to the image to generate a caption for or the image itself.
            max_length (int): The maximum length of the caption to generate.
            temperature (float): The temperature for the softmax.
            top_k (int): The number of top k tokens to consider for the top-k sampling.
            top_p (float): The top p value for the top-p sampling.
            prefix_text (str): The prefix text for the caption.
            min_length (int): The minimum length of the caption.
            k (int): The number of top captions generated by clip to consider.
            plot_individual (bool): Whether to plot the individual XAI gradcam maps.
            combined_threshold (float): The threshold for the combined XAI gradcam maps.
            combined_alpha (float): The alpha value for the combined XAI gradcam map.
            normalize_alpha_scaling (bool): Whether to normalize the alpha scaling for the combined XAI gradcam map.
        """

        # Get Image Prefix Embedding 
        # Use the CLIP wrapper to process the image and get its embedding ('prefix').
        # We also return the probabilities, the combined plot image and the individual plot images for 
        # XAI purposes using Grad-CAM.
        clip_probs, image_features, combined_plot_image, individual_plot_images = self.clip_wrapper.visualize(
            image=image,
            k=k,    
            plot_individual=plot_individual,
            combined_threshold=combined_threshold,
            combined_alpha=combined_alpha,
            normalize_alpha_scaling=normalize_alpha_scaling
        )

        # Copy the image features into a prefix variable
        prefix = image_features

        # Prepare Model and Inputs
        # Set the model to evaluation mode. This disables dropout and batch normalization updates,
        # ensuring consistent behavior during inference.
        self.eval()

        # Set a fixed random seed for reproducibility of the generation process.
        # Note: This is reset if the retry logic is triggered.
        torch.manual_seed(42)

        # Ensure the image prefix embedding is on the correct device (e.g., GPU).
        prefix = prefix.to(self.device)

        # If the prefix is a 1D tensor, add a batch dimension (unsqueeze) to make it 2D [1, embedding_dim].
        if prefix.ndim == 1:
            prefix = prefix.unsqueeze(0)

        # Prepare Text Prefix
        # Encode the initial text prompt (e.g., "A plant disease description: ") into token IDs.
        # The result is a tensor of shape [1, sequence_length], moved to the correct device.
        prefix_tokens = self.tokenizer.encode(prefix_text, return_tensors="pt").to(self.device)

        # Initialize the sequence of generated tokens with the prefix tokens.
        generated_tokens = prefix_tokens

        # Define Tokens to Avoid Early
        # Create a list of token ID lists that should not be generated before reaching `min_length`.
        # This typically includes the End-of-Sequence (EOS) and Padding (PAD) tokens.
        bad_words_ids = [[self.tokenizer.eos_token_id], [self.tokenizer.pad_token_id]]

        # Manual Generation Loop
        try:

            # Loop to generate tokens one by one, up to the specified `max_length`.
            for i in range(max_length):

                # Get Model Output
                # Pass the current sequence (`generated_tokens`) and the image prefix (`prefix`)
                # to the model's forward pass to get predictions for the next token.
                # `mask` is None, assuming the model handles attention masking internally based on token IDs.
                outputs = self(
                    tokens=generated_tokens,
                    prefix=prefix,
                    mask=None
                )

                # Extract Next Token Logits
                # Get the logits (raw scores) for the *last* token position in the sequence.
                # Shape: [batch_size, vocab_size] (batch_size is 1 here).
                next_token_logits = outputs.logits[:, -1, :]

                # Apply Temperature Scaling
                # Scale the logits by the temperature. Lower temp -> sharper distribution (less random),
                # higher temp -> flatter distribution (more random).
                # Avoid division by zero if temperature is very small.
                if temperature > 1e-5:
                        next_token_logits = next_token_logits / temperature

                # Prevent Bad Tokens (Early EOS/PAD)
                # If the generated sequence length is less than `min_length`,
                # set the probability of generating EOS or PAD tokens to effectively zero (-infinity logits).
                if i < min_length:

                    # for each bad word id list in the bad words ids list:
                    for bad_ids_list in bad_words_ids:

                        # Ensure indices are valid before applying penalty
                        valid_indices = [idx for idx in bad_ids_list if idx < next_token_logits.shape[-1]]

                        # if there are valid indices:
                        if valid_indices:

                            # set the logits of the valid indices to -infinity: 
                            # this effectively prevents the model from generating the bad tokens.
                            next_token_logits[:, valid_indices] = -float('inf')


                # Apply Top-K Filtering
                # Top-K filtering is a technique used to limit the number of tokens that can be generated.
                # It's used to prevent the model from generating a large number of tokens that are not likely to be correct.
                # Keep only the 'top_k' most likely tokens and set the logits of others to -infinity.
                # If top_k is 0, this step is skipped.
                if top_k > 0:

                    # Get the logits of the k-th most likely token.
                    top_k_values, _ = torch.topk(next_token_logits, top_k)
                    
                    # Get the k-th value and keep dimension.
                    kth_logit = top_k_values[..., -1, None]

                    # Set logits below the k-th logit to -infinity.
                    next_token_logits[next_token_logits < kth_logit] = -float('inf')

                # Apply Top-P (Nucleus) Filtering
                # Top-P filtering after top-k filtering is used to, again, limit the number of tokens that can be generated.
                # It considers the cumulative probability of the tokens, so it's more flexible than top-k filtering.
                # Keep the smallest set of tokens whose cumulative probability exceeds `top_p`.
                # Set the logits of tokens outside this nucleus to -infinity.
                # If top_p is 1.0 or more, this step is skipped.
                if top_p < 1.0:

                    # Sort logits in descending order.
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    
                    # Calculate cumulative probabilities.
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                    # Create a mask for tokens to remove (those outside the nucleus).
                    # Shift the mask right to keep the first token that exceeds the threshold.
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0 # Always keep the most probable token

                    # Scatter the removal mask back to the original logit order.
                    # The scatter function is used to scatter the removal mask back to the original logit order.
                    # This is done because the removal mask is in the sorted order of the logits, so we need to 
                    # scatter it back to the original order of the logits.
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    
                    # Apply the mask, setting logits to -infinity.
                    next_token_logits[indices_to_remove] = -float('inf')

                # Sample Next Token
                # Convert the final filtered logits to probabilities using softmax.
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                # Sample one token based on the calculated probabilities.
                next_token = torch.multinomial(probs, num_samples=1)

                # Check for EOS
                # Get the numerical ID of the sampled token.
                token_id = next_token.item()

                # If the EOS token is generated *and* we have reached the minimum length, stop generation.
                if token_id == self.tokenizer.eos_token_id and i >= min_length:
                    self.logger.debug(f"EOS token generated at step {i+1}, stopping generation.")
                    
                    # Exit the generation loop
                    break 

                # Append Token to Sequence after checking for EOS
                # Concatenate the newly generated token to the sequence for the next iteration.
                generated_tokens = torch.cat((generated_tokens, next_token), dim=1)

                # Optional Debugging: Log the generated token
                # self.logger.debug(f"Step {i+1}: Token ID: {token_id}, Token: '{self.tokenizer.decode([token_id])}'")


            # Decode and Post-process
            # Decode the final sequence of token IDs back into a string.
            # 'skip_special_tokens=True' removes tokens like EOS, PAD automatically.
            caption = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

            # Remove the initial prefix text from the beginning of the caption.
            caption = caption.replace(prefix_text, "").strip()

            # Validate Caption and Retry Logic
            # Check if the generated caption is potentially invalid (empty, too short, only punctuation).
            is_invalid = (
                not caption or                   # - Is empty
                caption.isspace() or             # - Contains only whitespace
                len(caption) < min_length or     # - Shorter than minimum required length
                all(c in '?!.,;:' for c in caption) # - Consists only of basic punctuation
            )

            if is_invalid:

                # Log a warning about the invalid caption.
                self.logger.warning("Generated invalid caption, retrying with different seed and slightly increased temperature...")
                
                # Attempt to generate again by recursively calling the function with:
                # - A new random seed (based on the current system time/state).
                # - Slightly increased temperature to encourage more diverse output.
                # Note: This could potentially lead to infinite recursion if the model consistently fails.
                # Consider adding a retry limit if this becomes an issue.
                torch.manual_seed(torch.seed() + 1) # Set a new seed
                return self.generate_caption(
                    image_path=image_path,
                    max_length=max_length,
                    temperature=temperature * 1.1, # Increase temperature slightly
                    top_k=top_k,
                    top_p=top_p,
                    prefix_text=prefix_text,
                    min_length=min_length
                )

            # Final Cleanup
            # Remove any non-printable characters that might have slipped through.
            caption = ''.join(c for c in caption if c.isprintable())

            # Return the final, cleaned caption alongside the image visual explanations.
            return clip_probs, caption, combined_plot_image, individual_plot_images

        # Error Handling
        except Exception as e:

            # Log any exception that occurs during the generation loop.
            self.logger.error(f"Error during manual caption generation: {str(e)}", exc_info=True)
            
            # Return an error message string.
            # import traceback # Keep traceback import local if only used here
            # traceback.print_exc()
            return f"[Error during generation: {str(e)}]"

    # Function to train the model on the dataset.
    def train_model(self, dataset_path, output_dir, epochs, batch_size=32, learning_rate=2e-5, warmup_steps=5000):

        # Setup training configuration
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging to display informations
        self.log_file = self.output_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)

        # Load the dataset object from the dataset path
        dataset = Dataset(dataset_path)
        
        # Setup training components
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda step: min(1.0, float(step) / warmup_steps))
        
        # Starting the logging of the trainings
        self.logger.info("Starting training...")
        
        # For the number of requested epochs needed
        for epoch in range(epochs):

            # Put the model in train mode: that means that the model will be trained on the data
            # This is necessary for the backpropagation and gradient descent to work.
            self.train()

            # We start with a loss of 0 and build the tqdm module for the progress bar
            epoch_loss = 0
            progress = tqdm(total=len(self.dataloader), desc=f'Epoch {epoch}')
            
            # For each batch in the dataloader: The batch is a tuple of embeddings and descriptions.
            # The embeddings are the CLIP embeddings of the images and the descriptions are their captions.
            for batch_idx, (embeddings, descriptions) in enumerate(self.dataloader):
                embeddings = embeddings.to(self.device)
                
                # Tokenize embeddingg descriptions: with the tokenizer we convert the descriptions into tokens.
                tokenized = self.tokenizer(
                    descriptions,
                    padding="max_length",
                    truncation=True,
                    max_length=77,
                    return_tensors="pt"
                )
                
                # Convert the tokens into a tensor and move it to the device.
                tokens = tokenized.input_ids.to(self.device)

                # Create a mask for the tokens: this is used to mask the padding tokens.
                # This is necessary for the model to ignore the padding tokens, so that the model 
                # doesn't learn to repeat them.
                mask = tokenized.attention_mask.to(self.device)
                
                # Forward pass: we pass the tokens and the embeddings to the model.
                # The model then returns the logits for the next token.
                outputs = self(tokens, embeddings, mask, labels=tokens)

                # Calculate the loss: the loss is the difference between the predicted tokens and the actual tokens.
                loss = outputs.loss

                # Add the loss to the epoch loss.
                epoch_loss += loss.item()
                
                # Backward pass: we calculate the gradient of the loss with respect to the model parameters.
                loss.backward()

                # Update the model parameters.
                self.optimizer.step()

                # Update the learning rate.
                self.scheduler.step()

                # Zero the gradients.
                self.optimizer.zero_grad()
                
                # Update the progress bar.
                progress.set_postfix({"loss": loss.item()})
                progress.update()
            
            # Close the progress bar.
            progress.close()
            
            # Log epoch results
            avg_loss = epoch_loss / len(self.dataloader)
            self.logger.info(f"Epoch {epoch} - Average Loss: {avg_loss:.4f}")
            
            # Save checkpoint if the epoch is a multiple of 5.
            if (epoch + 1) % 5 == 0:
                self.save_pretrained(epoch, avg_loss)
        
        return self
    
    def save_pretrained(self, epoch, loss):

        # Build the path to the checkpoint file.
        checkpoint_path = self.output_dir / f"clipcap_epoch_{epoch}.pt"

        # Save the model checkpoint.
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, checkpoint_path)

        # Log the checkpoint save.
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def from_pretrained(self, model_path):

        # If the device is not set, set it to the device of choice.
        if self.device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)

        # Load the model state dictionary.
        self.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        # Return the model.
        return self


