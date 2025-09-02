# File to wrap models in class to be able to use them in a more convenient way.
# You can add your own models to this file and then load them in the VLM.
import os
import io
import cv2 
import clip
import torch
import logging
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import torch.nn.functional as F # For ReLU
import matplotlib.patches as mpatches # Needed for legend

from PIL import Image
from tqdm import tqdm
from utils.summarizer import synthesize
from utils.plot import individual_map, combined_map


# We create a wrapper class and inherit from torch.nn.Module to be able to use pytorch funcionality
# if needed. 
class CLIPW(torch.nn.Module):

    # Initialize the class with the model and preprocess module
    def __init__(self, model_name="ViT-B/32", captions=None, device=None):
        super().__init__()

        # Initialize the device
        self.device = device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize the model and the preprocess module
        self.model, self.preprocess = clip.load(model_name, device=self.device)

        # If knowledge is provided, we encode the text embeddings to avoid doing it
        # at each prediction. Knowledge it's basically a list of captions.
        # We also save the captions to be able to use them later.
        if captions is not None:
            self.captions = captions
            self.text_tokens = clip.tokenize(captions).to(self.device)
        else:
            self.captions = None
            self.text_tokens = None

        # Set up logging for the wrapper class
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    # Function to load an image and preprocess it using the preprocess module
    def load_image(self, image_path):

        # Load the image with PIL and convert it to RGB
        image = Image.open(image_path).convert("RGB")

        # Preprocess the image and return it as a tensor loaded on the device
        return self.preprocess(image).unsqueeze(0).to(self.device)
    
    # Function to solely preprocess the image
    def preprocess_image(self, image):

        # Preprocess the image and return it as a tensor loaded on the device
        return self.preprocess(image).unsqueeze(0).to(self.device)
    
    # UPDATED: Internal helper function now returns tensor and the intermediate PIL image if requested
    def _prepare_image_tensor(self, image, return_raw_image=False):
        """
        Loads an image if a path is provided, converts NumPy arrays (BGR to RGB),
        ensures PIL Images are RGB, preprocesses, and returns both the
        intermediate PIL Image and the final tensor.

        Args:
            image (str | PIL.Image.Image | np.ndarray):
                The image input, which can be a file path, a PIL Image, or a NumPy array.

        Returns:
            tuple[PIL.Image.Image, torch.Tensor]:
                A tuple containing:
                - image_pil: The intermediate PIL Image object (in RGB format).
                - image_tensor: The preprocessed image tensor ready for the model.

        Raises:
            TypeError: If the input type is not supported.
            FileNotFoundError: If the path does not exist (when input is str).
            Exception: For other image loading/processing errors.
        """

        # Initialize the PIL image
        image_pil = None

        # If the input is a file path, we open and convert directly. This creates the PIL image.
        if isinstance(image, str):

            # Input is a file path
            try:
            
                # Open and convert directly. This creates the PIL image.
                image_pil = Image.open(image).convert("RGB")
            
            # If the file is not found, we log an error and raise an exception
            except FileNotFoundError:
                self.logger.error(f"Image file not found at path: {image}")
                raise
            
            # If there is an error, we log it and raise an exception
            except Exception as e:
            
                # Log the error
                self.logger.error(f"Error opening or converting image file {image}: {e}")

                # Raise an exception
                raise
            
        # If the input is a NumPy array, we convert it to a PIL image
        elif isinstance(image, np.ndarray):

            # Input is a NumPy array (assume BGR from cv2)
            try:
                # Convert the NumPy array to a PIL image
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Create the PIL image
                image_pil = Image.fromarray(image_rgb)

            # If there is an error, we log it and raise an exception
            except Exception as e:
                self.logger.error(f"Error converting NumPy array to PIL Image: {e}")
                raise
            
        # If the input is a PIL image, we assure to convert it to RGB
        elif isinstance(image, Image.Image):

            # Input is already a PIL Image
            try:
                # convert("RGB") returns a converted copy if not already RGB.
                image_pil = image.convert("RGB")
            
            except Exception as e:
                self.logger.error(f"Error converting PIL Image to RGB: {e}")
                raise
        else:
            # Unsupported type
            raise TypeError(f"Unsupported image input type: {type(image)}. Expected str, PIL.Image, or np.ndarray.")

        # Finally we preprocess the PIL image and return it as a tensor loaded on the device
        try:
            # Preprocess the PIL image and return it as a tensor loaded on the device
            image_tensor = self.preprocess(image_pil).unsqueeze(0).to(self.device)

            # Return both the PIL image and the tensor if requested
            if return_raw_image:
                return image_tensor, image_pil
            
            # Return only the tensor if not requested
            else:
                return image_tensor
        
        except Exception as e:
            self.logger.error(f"Error during CLIP preprocessing: {e}")
            raise

    # Function to extract solely the image embedding
    def image_embedding(self, image_path):

        # Load and preprocess the image
        image_tensor = self.load_image(image_path)

        # Get the image embedding
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)

        # Return the image embedding
        return image_features
    
    # Function to extract solely the text embedding
    def text_embedding(self, captions):

        # Tokenize the captions
        text_tokens = clip.tokenize(captions).to(self.device)

        # Get the text embedding
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)

        # Return the text embedding
        return text_features

    # Forward function to pass the image through the model
    def predict(self, image, captions=None, return_img_emb=False, return_text_emb=False):

        # If no captions are provided, we check if already have the text features from 
        # a previous injection of caption knowledge during the initialization phase
        # If not, we raise an error and exit.
        if captions is None and self.text_tokens is None:
            raise ValueError("No captions provided and no knowledge injected.")


        # Prepare the image tensor
        image_tensor, _ = self._prepare_image_tensor(image)

        # If captions are provided, we encode the text embeddings
        if captions is not None:
            text_tokens = clip.tokenize(captions).to(self.device)

        # If no captions are provided, we use the saved text tokens from the initialization
        else:
            text_tokens = self.text_tokens

        # Dummy variables to store the features
        image_features = None
        text_features = None    
        
        # With torch.no_grad() we ensure that no gradients are calculated during the forward pass
        with torch.no_grad():

            # If return_img_emb is True, we get the image embedding
            if return_img_emb:
                image_features = self.model.encode_image(image_tensor)

            # If return_text_emb is True, we get the text embedding
            if return_text_emb:
                text_features = self.model.encode_text(text_tokens)

            # We pass the image tensor and the tokenized text through the model to get the logits
            # In this case, the logits are the similarity between the image and the text
            # And we want to get the probabilities of the image being similar to each of the captions.
            # Basically, 
            # - logits_per_image = image_features×text_features^T
            # - logits_per_text = text_features×image_featuresT
            logits_per_image, logits_per_text = self.model(image_tensor, text_tokens)

            # We calculate the probabilities by applying a softmax to the logits_per_image.
            # This is due to the fact that the logits are the similarity between the image and the text
            # And we want to get the probabilities of the image being similar to each of the captions.
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        # Return the probabilities
        return probs, image_features, text_features
        
    # Function that extracts the top K captions and their probabilities from the model output
    def top_k_predictions(self, probs, captions=None, k=5):

        # If no captions are provided and we don't have captions saved, we raise an error and exit.
        if captions is None and self.captions is None:
            raise ValueError("No captions provided and no knowledge injected.")
        
        # If no captions are provided, we use the saved captions
        if captions is None:
            captions = self.captions

        # Extract the indices of the top N probabilities.
        # Probs[0] is the first image in the batch, so we get the probabilities for that image.
        # We get the indices of the top N probabilities in descending order.
        # ::-1 is used to reverse the order of the indices.
        top_indices = probs[0].argsort()[-k:][::-1] 

        # Create a list of tuples with the caption and the probability
        top_captions = [(captions[i], probs[0][i]) for i in top_indices]

        # Normalize the top 5 probabilities to sum to 1
        top_probs = [prob for _, prob in top_captions]
        total_prob = sum(top_probs)

        # We create a list of tuples with the caption and the normalized probability
        normalized_captions = [(caption, prob / total_prob) for caption, prob in top_captions]

        # Return the normalized captions
        return normalized_captions
    
    # Function to get a synthesized caption from the model output
    def synthesize_caption(self, prob, captions=None):

        # If no captions are provided and we don't have captions saved, we raise an error and exit.
        if captions is None and self.captions is None:
            raise ValueError("No captions provided and no knowledge injected.")
        
        # If no captions are provided, we use the saved captions
        if captions is None:
            captions = self.captions

        # We call the best_predictions function to get the top caption and its probability
        top_predictions = self.top_k_predictions(prob, captions, k=5)

        # Call the synthesize function from the summarizer module
        return synthesize(top_predictions)
    
    # Function to preprocess the dataset into a dataset of embeddings and captions.
    # This will be needed for the training of the captioner model to avoid computing
    # the embeddings and the captions at each epoch of training, saving time and memory.
    def preprocess_dataset(self, root_dir: str, output_dir: str, batch_size: int = 32):

        # Start logging 
        self.logger.info("Starting dataset preprocessing...")
        
        # We check if a dataset containing preprocessed embeddings already exists
        try:
            # Check if dataset already exists
            if os.path.exists(output_dir):
                self.logger.info(f"A Dataset embeddings folder already exists in {output_dir}. "
                "Delete any existing folder before trying again.")
                return
            
        # Basic error handling for the given folder
        except Exception as e:
            self.logger.error(f"Error checking dataset existence: {str(e)}")
            return

        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize lists to store image IDs and captions
        image_ids = []
        captions = []
        failed_images = []
        
        # Get all image paths
        image_paths = []

        # The structure of the root directory need to be as the following:
        # root_dir/
        #   class1/
        #     img1.jpg
        #     img2.png
        #     ...
        #   class2/
        #     img1.jpeg
        #     img2.jpg
        #     ...
        #   classK/
        #     img1.jpg
        #     img2.jpg
        #     ...
        # So we iterate over all the class folders in the root directory and get all the image paths.
        for class_folder in os.listdir(root_dir):

            # Get the path of the class folder
            class_path = os.path.join(root_dir, class_folder)

            # If the class folder is not a directory, we skip it
            if not os.path.isdir(class_path):
                continue

            # Iterate over all the images in the class folder
            for img_name in os.listdir(class_path):

                # If the image ends with a compatible format, we add it to the list of image paths
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):

                    # Get the path of the image
                    img_path = os.path.join(class_path, img_name)

                    # Add the image path to the list of image paths
                    image_paths.append(img_path)
        
        # Pre processing steps: get the probabilities and the image embeddings for each image
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing images"):

            # Get the batch of image paths  
            batch_paths = image_paths[i:i + batch_size]

            # Iterate over all the images in the batch
            for img_path in batch_paths:

                # Try to get the probabilities and the image embeddings for the image   
                try:

                    # Get the image ID: we use the index of the image in the batch as the image ID
                    # and we format it to be 8 digits long.
                    img_id = f"emb_{len(image_ids):08d}"
                    
                    # Get probabilities and image embeddings using the predict function
                    probs, img_embedding, _ = self.predict(img_path, return_img_emb=True)
                    
                    # Generate caption
                    caption = self.synthesize_caption(probs)
                    
                    # Save embedding in the output directory as a .pt file (pytorch file)
                    torch.save(img_embedding, os.path.join(output_dir, f"{img_id}.pt"))
                    
                    # Add the image ID and the caption to the lists
                    image_ids.append(img_id)
                    captions.append(caption)
                    
                # If there is an error, we log it and add the image path to the list of failed images
                except Exception as e:

                    # Log the error
                    self.logger.error(f"Error processing {img_path}: {str(e)}")

                    # Add the image path to the list of failed images
                    failed_images.append(img_path)
        
        # Save mapping: we save the image IDs and the captions in a csv file
        # using a pandas dataframe.
        df = pd.DataFrame({
            'embedding_id': image_ids,
            'caption': captions
        })

        # Save the dataframe to a csv file
        df.to_csv(os.path.join(output_dir, 'caption_mapping.csv'), index=False)
        
        # Save failed images list: we save the list of failed images in a txt file  
        # in the same directory as the output directory.
        if failed_images:

            with open(os.path.join(output_dir, 'failed_images.txt'), 'w') as f:
                f.write('\n'.join(failed_images))
        
        # Log the number of images processed successfully and the number of failed images
        self.logger.info(f"Processed {len(image_ids)} images successfully")
        self.logger.info(f"Failed to process {len(failed_images)} images")

    # Visualize Grad-CAM explanations for each of the top K captions wanted.
    # - If k=1, saves a single Grad-CAM map for the top caption.
    # - If k>1, saves a combined overlay map and optionally individual maps
    #   for each of the top K captions.
    # Probabilities are included in the plots.
    def visualize(self,
            image,
            k=1,
            layer_index=-2,
            plot_individual=True,
            combined_threshold=0.5,
            combined_alpha=1.0,
            normalize_alpha_scaling=False):

        # Explanation tof the parameters of the function since it's a bit complex:
        """
        Args:
            image_path (str): Path to the input image.
            k (int): The number of top predictions to visualize (default: 1).
            layer_index (int): Index of the ResidualAttentionBlock layer to target. Defaults to -2.
            plot_individual (bool): If True and k > 1, save individual Grad-CAM maps for each rank.
            combined_threshold (float): Threshold for the combined overlay (0-1, used if k > 1).
            combined_alpha (float): Alpha blending for the combined overlay (0-1, used if k > 1).
            normalize_alpha_scaling (bool): If True (default), normalize the top K probabilities
                                            and use them for alpha scaling in the combined plot.
                                            If False, use a constant alpha (`combined_alpha`)
                                            for all regions in the combined plot.
                                            The legend always shows original probabilities.
        """


        # Set the model to evaluation mode (disables dropout, batch norm updates)
        self.model.eval()

        # Check if captions are available:
        # Verify that captions and their tokenized representations exist in the wrapper instance
        # created at the initialization of the wrapper.
        if self.captions is None or self.text_tokens is None:
            
            # Log an error if captions haven't been initialized
            self.logger.error("Captions not initialized in the wrapper. Cannot perform Grad-CAM.")
            
            # Exit the function if captions are missing
            return
        
        # Check if the requested number of top predictions (k) is valid
        if k < 1:

            # Log an error if k is less than 1
            self.logger.error("k must be 1 or greater.")
            
            # Exit the function if k is invalid
            return
        
        # Check if the requested k exceeds the number of available captions
        if k > len(self.captions):
             
             # Log a warning and adjust k to the maximum available captions
             self.logger.warning(f"Requested k={k}, but only {len(self.captions)} captions are available. Setting k={len(self.captions)}.")
             
             # Set k to the number of available captions
             k = len(self.captions)

        # Log the start of the Grad-CAM generation process
        # If there is an image path, log the info else log a generic message.
        if isinstance(image, str):
            self.logger.info(f"Generating Grad-CAM for top {k} caption(s) for: {image}")
        else: 
            self.logger.info(f"Generating Grad-CAM for top {k} caption(s) for the given image.")

        # Prepare image tensor and the intermediate PIL image
        image_tensor, image_pil = self._prepare_image_tensor(image, return_raw_image=True)

        # Identify Target Layer:
        # Get the visual part of the CLIP model
        visual_model = self.model.visual

        # Initialize the target layer variable to None: this will be used to store the target layer later.
        target_layer = None
        
        # Initialize the target block name for logging purposes
        target_block_name = "N/A"
        
        # Check if the visual model has a 'transformer' attribute and if that has 'resblocks'
        if hasattr(visual_model, 'transformer') and hasattr(visual_model.transformer, 'resblocks'):

            # Get the total number of residual blocks in the transformer
            num_blocks = len(visual_model.transformer.resblocks)

            # Check if the provided layer_index is out of bounds (positive or negative indexing)
            if abs(layer_index) > num_blocks or layer_index >= num_blocks:
                 
                 # Log an error if the layer index is invalid
                 self.logger.error(f"Invalid layer_index {layer_index}. Model only has {num_blocks} blocks (indices 0 to {num_blocks-1} or -1 to {-num_blocks}).")
                 
                 # Exit the function if the index is invalid
                 return
            
            # Start a try-except block to handle potential errors when accessing the layer
            try:

                # Access the specific residual block using the provided index
                target_layer = visual_model.transformer.resblocks[layer_index]

                # Set the target block name for logging
                target_block_name = f"ResidualAttentionBlock index {layer_index}"

                # Log the layer being used for Grad-CAM hooks
                self.logger.info(f"Using target layer for hooks: {target_block_name}")

            # Catch IndexError if the index is valid but still causes an error (should be rare after bounds check)
            except IndexError:
                 
                 # Log an error if the block cannot be accessed
                 self.logger.error(f"Could not access block at index {layer_index}.")
                 
                 # Exit the function
                 return
            
        # If the expected structure (transformer.resblocks) is not found
        else:

            # Log an error indicating the structure is different
            self.logger.error("Could not automatically determine target layer path (visual_model.transformer.resblocks).")
            
            # Exit the function
            return

        # Ensure correct image dtype (float32, but sometimes CLIP uses float16):
        # Start a try-except block to handle potential dtype errors
        try:
             
             # Convert the image tensor to the same data type as the weights of the first convolutional layer
             # This ensures compatibility for the forward pass later on 
             image_tensor = image_tensor.to(visual_model.conv1.weight.dtype)
        
        # Catch any exception during dtype conversion
        except Exception as e:
             
             # Log an error if dtype conversion fails
             self.logger.error(f"Error setting image tensor dtype: {e}")
             
             # Exit the function
             return

        # Step 1: Prediction Pass - Find Top K Captions:
        # Initialize lists to store the indices, captions, and probabilities of the top K predictions
        # Note: we already have the top K caption function in the wrapper, but the function just returns
        # the top K captions, not the indices and probabilities. So instead of computing the image twice
        # we just do it here with a custom logic.
        top_k_indices = []
        top_k_captions = []
        top_k_probs = []

        # We also initialize a list to store the normalized probabilities for the alpha scaling.
        # This is because we need to store the normalized probabilities to be able to plot the
        # combined Grad-CAM map later on.
        normalized_top_k_probs = []

        # Initialize the dictionary to store caption-probability pairs
        caption_probs_dict = {}

        # Start a try-except block for the prediction pass
        try:

            # Use torch.no_grad() to disable gradient calculations for this inference step
            with torch.no_grad():

                # Perform a forward pass with the image and all text tokens to get similarity logits
                logits_per_image, _ = self.model(image_tensor, self.text_tokens)

                # Apply softmax to the image logits to get probabilities for each caption
                probs = logits_per_image.softmax(dim=-1)

                # Create the caption-probability dictionary 
                # Assuming batch size is 1, get probabilities for the first (and only) image
                all_probs_list = probs[0].cpu().tolist()

                # Create the dictionary mapping each caption to its probability
                caption_probs_dict = {caption: prob for caption, prob in zip(self.captions, all_probs_list)}

                # Sort the dictionary by probability (descending)
                sorted_items = sorted(caption_probs_dict.items(), key=lambda item: item[1], reverse=True)
                caption_probs_dict = dict(sorted_items)

                # Get the top K probabilities and their corresponding indices from the probabilities tensor
                # .cpu() moves the tensor to the CPU before converting to lists/numpy
                top_probs_tensor, top_idx_tensor = probs.cpu().topk(k, dim=-1)

                # Convert the top K indices tensor to a list (index 0 because batch size is 1)
                top_k_indices = top_idx_tensor[0].tolist()

                # Convert the top K probabilities tensor to a list (index 0 because batch size is 1)
                top_k_probs = top_probs_tensor[0].tolist()

                # Retrieve the actual caption strings corresponding to the top K indices
                top_k_captions = [self.captions[i] for i in top_k_indices]

                # Once we have the top K probabilities, we normalize them to sum to 1.
                prob_sum = sum(top_k_probs)

                # This is to avoid division by zero or near-zero.
                if prob_sum > 1e-6:

                    # Normalize the probabilities: divide each probability by the sum of all probabilities.
                    normalized_top_k_probs = [p / prob_sum for p in top_k_probs]
                else:

                    # If sum is tiny, assign equal probability (or handle as error)
                    self.logger.warning("Sum of top K probabilities is near zero. Using equal normalized probabilities.")
                    normalized_top_k_probs = [1.0 / k] * k

                # print the normalized probabilities
                self.logger.info(f"Normalized probabilities: {normalized_top_k_probs}")

                # Log that the top K predictions have been identified
                self.logger.info(f"Top {k} predictions identified.")

                # Log whether normalization will be used for alpha scaling
                if normalize_alpha_scaling:
                    self.logger.info("Using NORMALIZED probabilities for combined plot alpha scaling.")
                else:
                    self.logger.info("Using CONSTANT alpha for combined plot alpha scaling.")

                # Loop through the top K results for debugging/logging purposes
                for i in range(k):

                    # Log the rank, caption, and probability for each top prediction (using debug level)
                    # Added normalized probability to the log message
                    self.logger.debug(
                        f"  Rank {i+1}: '{top_k_captions[i]}' "
                        f"(Prob: {top_k_probs[i]:.4f}, NormProb: {normalized_top_k_probs[i]:.4f})"
                    )


        # Catch any exception during the prediction pass
        except Exception as e:
            # Log an error if the prediction pass fails
            self.logger.error(f"Error during prediction pass to find top caption(s): {e}")
            
            # Exit the function
            return

        # Step 3: List to store computed CAM maps and their info:
        # Initialize an empty list to store results for successfully computed CAM maps
        # Each element will be a tuple: (caption, probability, cam_map_numpy_array)
        computed_results = []

        # Step 4: Loop through top K to compute Grad-CAM:
        # We need to compute the Grad-CAM for each of the top K captions.
        # Iterate through the ranks from 0 to k-1
        for rank in range(k):

            # Get the caption for the current rank
            current_caption = top_k_captions[rank]
            
            # Get the original index (in the full caption list) for the current rank
            current_index = top_k_indices[rank]
            
            # Get the probability for the current rank
            current_prob = top_k_probs[rank]
            
            # Get the tokenized representation for only the current caption
            current_text_tokens = self.text_tokens[current_index:current_index+1]

            # Log the start of Grad-CAM computation for the current rank
            self.logger.info(f"\n--- Computing Grad-CAM for Rank {rank+1}/{k}: '{current_caption}' ---")

            # Grad-CAM Pass (Inside Loop)
            # Initialize variables to store activations and gradients from hooks
            activations = None
            gradients = None

            # Define the forward hook function: We define it here because we need it to be a nonlocal variable
            # to be able to modify it in the hook function.
            # What is a hook function? It is a function that is called when a certain event happens in the model.
            # In this case, it is called when the forward pass is done: it captures the output tensor of the 
            # target layer during the forward pass, easy as that.
            def forward_hook(module, input, output): nonlocal activations; activations = output.detach()

            # Define the backward hook function: same logic as above.
            # It captures the gradient flowing back into the target layer during the backward pass
            def backward_hook(module, grad_input, grad_output):

                # Define the gradients variable as nonlocal to be able to modify it in the hook function.
                nonlocal gradients

                # Check if grad_output is a tuple (common case) and contains a tensor
                if isinstance(grad_output, tuple) and len(grad_output) > 0 and isinstance(grad_output[0], torch.Tensor):
                    
                    # Capture the detached gradient tensor
                    gradients = grad_output[0].detach()
                
                # If the gradient format is unexpected, set gradients to None
                else: gradients = None

            # Register the forward hook to the target layer, storing the handle to remove it later
            forward_handle = target_layer.register_forward_hook(forward_hook)

            # Register the backward hook to the target layer, storing the handle
            # Use register_full_backward_hook (backward hook is deprecated)
            backward_handle = target_layer.register_full_backward_hook(backward_hook)

            # Initialize the numpy CAM map variable for this iteration: We do this because
            # we need to store the numpy CAM map for this iteration to be able to plot it later. 
            # The numpy CAM map is a 2D array that contains the CAM map for the current caption.
            # Note that CAM means Class Activation Mapping.
            cam_map_np = None

            # Start a try-except-finally block for the Grad-CAM forward/backward pass
            try:

                # Ensure any previous gradients are cleared before this pass: we want only
                # the gradients of the current caption to be computed.
                self.model.zero_grad()
                
                # Perform a forward pass using the image and ONLY the current caption's tokens
                logits_per_image_current, _ = self.model(image_tensor, current_text_tokens)

                # The target score for backpropagation is the logit corresponding to the current image/caption pair
                # We take [0, 0] because batch size is 1 and we have only 1 caption token
                score = logits_per_image_current[0, 0]

                # Perform backpropagation starting from the target score
                # This calculates gradients throughout the network with respect to this score
                score.backward()

            # Catch any exception during the forward/backward pass
            except Exception as e:

                # Log an error if the pass fails for the current rank
                self.logger.error(f"Error during Grad-CAM forward/backward pass for Rank {rank+1}: {e}")
                
                # Ensure hooks are removed even if an error occurred before the finally block
                forward_handle.remove(); backward_handle.remove()
                
                # Skip adding results for this failed rank: we don't want to continue the loop.
                continue 

            # The finally block ensures that hooks are always removed, regardless of success or failure
            finally:

                # Remove the forward hook
                forward_handle.remove()
                
                # Remove the backward hook
                backward_handle.remove()

            # Step 5: Check if hooks captured data: After a forward/backward pass activations and gradients are captured.
            # Verify that both activations and gradients were successfully captured by the hooks
            if activations is None or gradients is None:
                
                # Log an error if either activations or gradients are missing
                self.logger.error(f"Failed to capture activations or gradients for Rank {rank+1}.")
                
                # Skip adding results for this failed rank
                continue # Move to the next iteration of the loop

            # Step 6: Check Sequence Length:
            # Check if the sequence length dimension of the activations is greater than 1.
            # The first token is often a class token, we need patch tokens for spatial CAM: ergo, seqlen > 1.
            # We exclude the first token from the activations. 
            # NOTE: spatial CAM is a CAM that is computed on the spatial dimensions of the image, not on the channel dimensions.
            if activations.shape[0] <= 1:
                
                # Log an error if the sequence length is too short (no patch tokens)
                self.logger.error(f"Target layer ({target_block_name}) output sequence length ({activations.shape[0]}) is too short for Rank {rank+1}. Expected > 1.")
                
                # Skip adding results for this failed rank
                continue # Move to the next iteration of the loop

            # Step 7: Compute Grad-CAM (Adapted for (seq, batch, embed) format): 
            # We need to adapt the CAM computation to the format of the activations tensor, otherwise 
            # a lot of indices will be out of bounds and generating errors. 
            # Start a try-except block for the detailed CAM computation steps
            try:

                # Get the dimensions of the activations tensor: sequence length, batch size, embedding dimension
                seq_len, batch_size, embed_dim = activations.shape

                # Log a warning if the batch size is not 1, as the logic assumes it
                if batch_size != 1: self.logger.warning(f"Grad-CAM assumes batch size 1, got {batch_size}.")

                # Exclude the first token (usually CLS token) from activations to get patch activations
                patch_activations = activations[1:, :, :]
                
                # Exclude the first token's gradient to get patch gradients (CLS token is excluded as above)
                patch_gradients = gradients[1:, :, :]

                # Calculate the neuron importance weights (alpha) by averaging gradients over the embedding dimension
                # keepdim=True maintains the dimension for broadcasting
                alpha = patch_gradients.mean(dim=2, keepdim=True)

                # Weight the patch activations by the calculated importance weights (alpha)
                weighted_activations = patch_activations * alpha

                # Weighted activations are a tensor of shape (seq_len-1, batch_size, embed_dim).
                # Sum the weighted activations across the embedding dimension to get the raw CAM
                # We sum over the embedding dimension because we want to get a single value per patch.
                # (the embedding dimension is the number of channels in the activations tensor).
                raw_cam = weighted_activations.sum(dim=2)

                # Apply ReLU to the raw CAM, keeping only positive contributions:
                # This is done because we only want the positive contributions to the CAM, 
                # as negative contributions would cancel out the positive ones.
                cam = F.relu(raw_cam)

                # Calculate the number of patches (sequence length minus the CLS token)
                num_patches = seq_len - 1

                # Calculate the width of the patch grid (assuming it's square:
                # this is because the activations are a 2D grid of patches).
                width = int(num_patches**0.5)

                # Check if the number of patches forms a perfect square:
                # if not, we log a warning because the visualization will be distorted.
                if width * width != num_patches:

                    # Log a warning if the patches don't form a square grid
                    self.logger.warning(f"Cannot form square grid from {num_patches} patches for Rank {rank+1}. Visualization might be distorted.")
                
                # Reshape the CAM tensor into a 2D grid (width x width)
                # .squeeze(-1) removes the batch dimension (which should be 1)
                cam = cam.view(width, width, batch_size).squeeze(-1)

                # Check if the CAM tensor became empty after reshaping/squeezing (should not happen)
                if cam.numel() == 0: raise ValueError("CAM became empty after reshape/squeeze.")

                # Normalize the CAM map to the range [0, 1]
                # The formula is: (cam - cam.min()) / (cam.max() - cam.min())
                # This is done because we want to normalize the CAM map to the range [0, 1]
                # so that it can be displayed as a heatmap.
                                
                # Subtract the minimum value
                cam = cam - cam.min()

                # Get the maximum value
                cam_max = cam.max()
                
                # Divide by the maximum value, avoiding division by zero or near-zero
                if cam_max > 1e-8: # Use a small epsilon for stability
                    cam = cam / cam_max
                
                # If the max value is very close to zero, the map is essentially blank
                else:
                    
                    # Log a warning that the map will be blank
                    self.logger.warning(f"Grad-CAM heatmap maximum is close to zero for Rank {rank+1}. Map will be blank.")
                    
                    # Set the CAM to zeros to avoid potential issues later
                    cam = torch.zeros_like(cam)

                # Convert the final CAM tensor to a NumPy array and move it to the CPU
                # (We do this because we want to be able to plot the CAM map later).
                cam_map_np = cam.cpu().numpy()

                # Store the successful result with both original and normalized probability
                # Append the caption, original probability, normalized probability, and computed numpy CAM map
                computed_results.append((current_caption, current_prob, normalized_top_k_probs[rank], cam_map_np))

                # Log the successful computation for this rank
                self.logger.info(f"Successfully computed Grad-CAM map for Rank {rank+1}.")


            # Catch any exception during the CAM computation details
            except Exception as e:
                 
                 # Log an error if CAM computation fails for this rank
                 self.logger.error(f"Error computing Grad-CAM map details for Rank {rank+1}: {e}")
                 # Skip adding results for this failed rank
                 continue # Move to the next iteration of the loop
                
    
        # After the loop, we have computed the Grad-CAM for each of the top K captions.
        # Now we can plot the results.
        # Step 8: Plotting Results:
        # Check if any CAM maps were successfully computed and stored
        if not computed_results:

            # Log an error if no maps were computed
            self.logger.error("No Grad-CAM maps were successfully computed. No plots will be generated.")
            
            # Exit the function
            return

        # Get the actual number of successfully computed maps
        actual_k = len(computed_results)

        # If the number of successful maps is less than the requested k, log a warning
        if actual_k < k:
             
             # Log a warning if fewer maps were computed than requested
             self.logger.warning(f"Only {actual_k} out of {k} requested Grad-CAM maps were computed successfully.")

        # Plotting logic:

        # Init an array to store the Grad-CAM map
        grad_cam_map = []

        # Check if only one map was successfully computed (or if we only wanted one map since we got one caption)
        if actual_k == 1:

            # Handle the special case of plotting a single map
            # Unpack the results for the single map (caption, original_prob, norm_prob, cam_map)
            # We use original prob (not normalized) for single plot title/filename
            caption, prob, _, cam_map = computed_results[0] 

            # Start a try-except block for plotting the single map
            try:

                # Create a safe filename from the caption (alphanumeric characters and underscores)
                safe_caption = "".join(c if c.isalnum() else "_" for c in caption)[:30] # Limit length
                
                # Construct the output filename including rank, probability, and caption snippet
                output_filename = f"rank_1_prob_{prob:.3f}_{safe_caption}.png"
                
                # Call the helper function to plot the single Grad-CAM overlay
                grad_cam_map.append(individual_map(image_pil, cam_map, caption, probability=prob))
                
                # Log the path where the visualization was saved
                self.logger.info(f"Grad-CAM visualization computed successfully.")
            
            # Catch any exception during plotting or saving the single map
            except Exception as e:
                
                # Log an error if plotting/saving fails
                self.logger.error(f"Error plotting/saving Grad-CAM map for the top caption: {e}")

        # If more than one map was successfully computed (actual_k > 1)
        else: 

            # Check if individual plots were requested (we want to plot each map individually)
            # If not we just plot the combined overlay map and skip entirely the individual maps.
            if plot_individual:
                
                # Log the start of plotting individual maps
                self.logger.info(f"\n--- Plotting Individual Grad-CAM Maps (Top {actual_k}) ---")
                
                # Iterate through the computed results
                for i in range(actual_k):
                    
                    # Unpack the results for the current rank (caption, original_prob, norm_prob, cam_map)
                    # As above, we use original prob for individual plot title/filename
                    caption, prob, _, cam_map = computed_results[i] 
                    
                    # Start a try-except block for plotting this individual map
                    try:
                        
                        # Create a safe filename snippet from the caption
                        safe_caption = "".join(c if c.isalnum() else "_" for c in caption)[:30]
                        
                        # Construct the output filename including rank, probability, and caption snippet
                        # Use zero-padding for rank number (e.g., rank_01, rank_02)
                        output_filename = f"individual_rank_{i+1:02d}_prob_{prob:.3f}_{safe_caption}.png"
                        
                        
                        # Call the helper function to plot the single Grad-CAM overlay, adding rank info to the title
                        grad_cam_map.append(individual_map(image_pil, cam_map, f"Rank {i+1}: {caption}", probability=prob))
                        
                        # Log the path where the individual map was saved
                        self.logger.info(f"Individual map for Rank {i+1} computed successfully.")
                    
                    # Catch any exception during plotting or saving this individual map
                    except Exception as e:
                        
                        # Log an error if plotting/saving fails for this rank
                        self.logger.error(f"Error plotting/saving individual Grad-CAM map for Rank {i+1}: {e}")

            
            # Once the individual maps are plotted (or not, depending on the plot_individual flag),
            # we plot the combined overlay map.
            # Log the start of generating the combined overlay
            self.logger.info(f"\n--- Generating Combined Grad-CAM Overlay (Top {actual_k}) ---")

            # Start a try-except block for generating the combined plot
            try:
                
                # Computed results is a list of tuples, each containing:    
                # - caption
                # - original probability
                # - normalized probability
                # - numpy CAM map

                # We unpack the computed results into separate lists for the plotting function. 
                valid_captions = [res[0] for res in computed_results]

                # We unpack the original probabilities.
                valid_original_probs = [res[1] for res in computed_results]

                # We unpack the normalized probabilities.
                valid_normalized_probs = [res[2] for res in computed_results]

                # We unpack the numpy CAM maps.
                valid_maps = [res[3] for res in computed_results]

                # Determine which probabilities to pass for scaling (None if flag is False)
                probs_for_scaling = valid_normalized_probs if normalize_alpha_scaling else None

                # Call the helper function
                grad_cam_combined_plot = combined_map(
                    original_img=image_pil,
                    captions=valid_captions,
                    cam_maps=valid_maps,
                    probabilities_for_alpha=probs_for_scaling,
                    original_probabilities=valid_original_probs,
                    k=actual_k,
                    threshold=combined_threshold,
                    alpha=combined_alpha,
                    normalize_alpha_scaling=normalize_alpha_scaling
                )

                # Log the path where the combined overlay was saved
                self.logger.info(f"Combined Grad-CAM overlay computed successfully.")
            
            # Catch any exception during the generation or saving of the combined plot
            except Exception as e:
                
                # Log an error if the combined plot generation fails
                self.logger.error(f"Error generating combined Grad-CAM overlay: {e}")

        # Log the completion of the Grad-CAM visualization process
        self.logger.info(f"\nFinished generating Grad-CAM visualizations.")

        # Before returning we need to convert the image tensor to image features for further use
        image_features = self.model.encode_image(image_tensor)

        # Return caption_probs_dict, image features, grad_cam_combined_plot, grad_cam_map
        return caption_probs_dict, image_features, grad_cam_combined_plot, grad_cam_map

