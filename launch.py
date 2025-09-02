"""
Gradio application for interactive caption generation and explainability
========================================================================

This script launches a Gradio interface that allows a user to:

- Upload an image (PIL)
- Configure caption generation parameters (temperature, prefix text, lengths)
- Configure explainability parameters for Grad-CAM overlays (alpha, normalization)
- Run inference with a trained ClipCaptioner model
- Visualize:
  - Combined Grad-CAM overlay (top-k prompts)
  - Individual Grad-CAM overlays (per prompt)
  - Top-5 probabilities (prompt → probability)
  - Final generated caption

High-level flow
---------------
1) Attempt to load a trained checkpoint into `ClipCaptioner`.
2) On user request, save the uploaded image temporarily and call `model.generate(...)`.
3) Post-process outputs for safe/UI-friendly rendering (types, lists, strings).
4) Display outputs and clean up temporary artifacts.

Notes
-----
- The model path can be changed via `MODEL_PATH`.
- The UI runs with `share=True` to optionally expose a temporary public URL.
- All exceptions are caught and rendered as helpful messages in the UI.
"""

import gradio as gr
import os
from PIL import Image
from models.vlm import ClipCaptioner  # Ensure this import path is correct for your project structure
import traceback  # For detailed error logging

# --- Configuration -----------------------------------------------------------------------------
# You can make these configurable via environment variables or CLI arguments if needed.
# - MODEL_PATH: path to a previously trained checkpoint to load for inference
# - TEMP_INPUT_FILENAME: ephemeral path used to persist the uploaded image to disk for processing
MODEL_PATH = "train/checkpoints/clipcap_epoch_9.pt"
TEMP_INPUT_FILENAME = "temp_gradio_input.jpg"  # Temporary file for processing

# --- Model Loading -----------------------------------------------------------------------------
# We attempt to initialize and load the ClipCaptioner model from a checkpoint. Any error is stored
# into `model_load_error` and the UI will show a helpful message to the user.
model = None
model_load_error = None
print("Attempting to load VLM model...")

# Try to load the model from the given path.
try:
    if os.path.exists(MODEL_PATH):

        # Initialize the VLM (OPT-125M + Mapper + CLIP wrapper)
        model = ClipCaptioner()

        # Load trained weights (mapper + LM states). CLIP is frozen by design.
        model = model.from_pretrained(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}.")

    else:

        # Could not find the model file at the given path.
        model_load_error = f"Model file not found at {MODEL_PATH}. Please check the path."
        print(model_load_error)

except Exception as e:

    # Persist full traceback for debugging; expose a concise version to the UI
    model_load_error = f"Error loading model: {e}\n{traceback.format_exc()}"
    print(model_load_error)
    model = None  # Ensure model is None if loading failed

# --- Processing Function -----------------------------------------------------------------------
def predict_and_visualize(
    image_input_pil,
    combined_alpha_val,
    temperature_val,
    prefix_text_val,
    normalize_alpha_val
):
    """
    Runs the VLM model on the input image and returns results for Gradio display.

    Args:
        image_input_pil (PIL.Image.Image): Input image from Gradio (PIL Image instance).
        combined_alpha_val (float): Alpha value for combined Grad-CAM overlay (0-1).
        temperature_val (float): Sampling temperature for caption generation (>0).
        prefix_text_val (str): Text prompt prepended before generation (conditioning the LM).
        normalize_alpha_val (bool): If True, combined overlay alpha is scaled by normalized top-k
                                    probabilities; if False, a constant alpha is used.


    Returns:
        tuple: (combined_plot_PIL, [individual_plots_PIL], top5_text, caption)
               On errors, returns placeholders and descriptive error strings.
    """
    # Guard: model not loaded
    if model is None:
    
        # Return an error message.
        error_msg = f"Model could not be loaded. Please check logs. Error details: {model_load_error}"
        return None, [], error_msg, error_msg

    # Guard: image missing
    if image_input_pil is None:
         return None, [], "Please upload an image and click 'Generate'.", ""

    # Get the path to the temporary input file (this is where the image is saved to disk for processing)
    img_path = TEMP_INPUT_FILENAME
    try:
        
        # Persist the PIL image to disk for downstream processing APIs that expect a path
        image_input_pil.save(img_path)
        print(f"Input image saved temporarily to {img_path}")
        print(f"Using parameters: combined_alpha={combined_alpha_val}, temperature={temperature_val}, prefix='{prefix_text_val}', normalize_alpha={normalize_alpha_val}")

        # Run the model's generate function with user-provided parameters
        probs, caption, combined_plot, individual_plots = model.generate(
            image=img_path,
            combined_alpha=combined_alpha_val, # Use slider value
            prefix_text=prefix_text_val,       # Use textbox value
            temperature=temperature_val,       # Use slider value
            normalize_alpha_scaling=normalize_alpha_val, # Use checkbox value
            # --- Add other fixed parameters if needed ---
            min_length=10,
            max_length=77,
            top_k=50,
            top_p=0.9,
            k=5,
            plot_individual=True # Ensure this is True if you want individual plots
        )
        print("Model generation complete.")

        # --- Format Outputs --------------------------------------------------------------------

        # 1) Top 5 Probabilities (pretty-printed)
        top_5_text = "Top 5 Predicted Captions:\n"

        # If the probabilities are available and in the expected format, sort them in descending order and format them
        if probs and isinstance(probs, dict):
            
            # Sort probabilities in descending order and format them
            sorted_probs = sorted(probs.items(), key=lambda item: item[1], reverse=True)
            
            # Get the top 5 probabilities and their corresponding captions
            top_5 = sorted_probs[:5]

            # If the top 5 probabilities are available, format them
            if top_5:
                
                # Format the top 5 probabilities
                top_5_text += "\n".join([f"{i+1}. {cap} (Prob: {prob:.4f})" for i, (cap, prob) in enumerate(top_5)])
            else:
                
                # If no probabilities were returned by the model, add an error message
                 top_5_text += "No probabilities were returned by the model."
        else:
            
            # If the probabilities are not available or in the unexpected format, add an error message
            top_5_text += f"Probabilities not available or in unexpected format: {type(probs)}"

        # 2) Individual Plots Gallery (ensure valid PIL Images for Gradio Gallery)
        individual_plots_list = []

        # If the individual plots are available and in the expected format, add them to the list
        if individual_plots and isinstance(individual_plots, list):
            
            # Initialize a flag to check if valid plots were found
            valid_plots_found = False

            # Iterate over the individual plots
            for i, plot in enumerate(individual_plots):
                
                # If the plot is a PIL Image, add it to the list and set the flag to True
                if isinstance(plot, Image.Image):
                    individual_plots_list.append(plot)
                    valid_plots_found = True
                else:
                    
                    # If the plot is not a PIL Image, add an error message
                    print(f"Warning: Item at index {i} in individual_plots is not a PIL Image: {type(plot)}")

            # If no valid plots were found, add an error message
            if not valid_plots_found:
                 print("Warning: No valid PIL Images found in individual_plots.")
        elif individual_plots:

            # If the individual plots are not a list, add an error message
             print(f"Warning: individual_plots is not a list: {type(individual_plots)}")
        else:

            # If no individual plots were generated or returned, add an error message
             print("Warning: No individual plots were generated or returned.")


        # 3) Combined Plot (must be a PIL Image for Gradio Image component)
        if not isinstance(combined_plot, Image.Image):

            # If the combined plot is not a PIL Image, add an error message
            print(f"Warning: combined_plot is not a PIL Image: {type(combined_plot)}")
            combined_plot = None # Set to None if not an image

        # 4) Caption (ensure string)
        if not isinstance(caption, str):
            
            # If the caption is not a string, add an error message
            print(f"Warning: Generated caption is not a string: {type(caption)}")
            caption = str(caption) # Attempt to convert to string

        # Print the generated caption, formatted top 5 probabilities, and number of individual plots returned for gallery
        print(f"Generated Caption: {caption}")
        print(f"Formatted Top 5 Probs: {top_5_text}")
        print(f"Number of individual plots returned for gallery: {len(individual_plots_list)}")

        # Return all prepared UI artifacts
        return combined_plot, individual_plots_list, top_5_text, caption


    except Exception as e:
        
        # Get the error details
        error_details = traceback.format_exc()
        print(f"Error during prediction: {e}\n{error_details}")
        
        # Return empty list for the gallery on error
        # This is a tuple containing:
        # - None: The combined plot (None if there was an error)
        # - An empty list: Individual plots (empty list if there was an error)
        # - A string containing the error message: Error during processing
        # - A string containing the error message: Error: <error_message>
        return None, [], f"Error during processing: {e}", f"Error: {e}"
    
    finally:
    
        # Clean up the temporary image file so repeated runs don't collide on disk
        if os.path.exists(img_path):
    
            # Try to remove the temporary image file
            try:
                os.remove(img_path)
                print(f"Temporary file {img_path} removed.")
            # If there is an error removing the temporary image file, add an error message
            except OSError as e:
                print(f"Error removing temporary file {img_path}: {e}")


# --- Gradio Interface Definition ----------------------------------------------------------------
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    # App title and short instructions
    gr.Markdown("# VLM Prediction Visualization")
    gr.Markdown("Upload an image and adjust parameters to generate a caption and visualize attention maps.")

    with gr.Row():
        # Input Column
        with gr.Column(scale=1):
            # Image input (PIL). Gradio handles conversion from browser uploads.
            input_image = gr.Image(type="pil", label="Input Image")

            # --- Add New Input Controls ---
            # Collapsible section for generation/visualization parameters to keep the UI clean
            # Use Accordion for tidiness
            with gr.Accordion("Generation Parameters", open=False): 

                # Textbox for the caption prefix
                input_prefix_text = gr.Textbox(
                    label="Caption Prefix",
                    value="A plant disease description: ",
                    info="Text to prepend to the generated caption."
                )

                # Slider for the combined plot alpha
                input_combined_alpha = gr.Slider(
                    minimum=0.0, maximum=1.0, step=0.05, value=0.4,
                    label="Combined Plot Alpha",
                    info="Transparency of attention overlay on the combined plot."
                )

                # Slider for the temperature
                input_temperature = gr.Slider(
                    minimum=0.0, maximum=1.0, step=0.05, value=0.7,
                    label="Temperature",
                    info="Controls randomness in generation (higher is more random)."
                )
                
                # Checkbox for the normalize alpha scaling
                input_normalize_alpha = gr.Checkbox(
                    label="Normalize Alpha Scaling",
                    value=True, # Default value
                    info="Whether to normalize attention map scaling."
                )
            # --- End New Input Controls ---

            # Primary action button (disabled if model failed to load)
            generate_button = gr.Button("Generate Caption & Visualize", variant="primary", interactive=model is not None)

            # Surface loading errors directly in the UI if present
            if model_load_error:
                 gr.Markdown(f"<font color='red'>**Model Loading Error:** {model_load_error}</font>")

        # Output Columns (arranged in a 2x2 grid conceptually)
        with gr.Column(scale=2):
             with gr.Row():
                 # Top Row Outputs
                 with gr.Column():
                     output_combined_plot = gr.Image(label="Combined Attention Plot", interactive=False)
                     output_top5_probs = gr.Textbox(label="Top 5 Probabilities", lines=7, interactive=False) # Increased lines slightly
                 # Bottom Row Outputs
                 with gr.Column():
                     # Replace gr.Image with gr.Gallery for individual plots
                     # Gallery to display individual Grad-CAM overlays (if any)
                     output_individual_plots = gr.Gallery(
                         label="Individual Attention Plots",
                         interactive=False,
                         columns=2, # Adjust number of columns as desired
                         rows=1,    # Adjust number of rows as desired
                         object_fit="contain", # How images should fit in the grid
                         height="auto" # Adjust height if needed
                     )
                     output_caption = gr.Textbox(label="Generated Caption", interactive=False)


    # --- Event Handling ---
    # Wire the event: when the button is clicked, call predict_and_visualize with inputs,
    # route outputs to the appropriate UI components
    generate_button.click(
        fn=predict_and_visualize,
        # Add the new input components to the inputs list
        inputs=[
            input_image,
            input_combined_alpha,
            input_temperature,
            input_prefix_text,
            input_normalize_alpha
        ],
        outputs=[
            output_combined_plot,
            output_individual_plots,
            output_top5_probs,
            output_caption
        ]
    )

# --- Launch the App ----------------------------------------------------------------------------
if __name__ == "__main__":
    print("Launching Gradio Interface...")
    # share=True creates a public link (optional)
    # debug=True provides more detailed logs in console (optional)

    # --- EDIT: Use Gradio's share feature for public access ---
    # Setting share=True will generate a temporary public URL (e.g., *.gradio.live)
    # This is useful if you cannot directly access the server's IP/port.
    print("Gradio will generate a public share link shortly.")
    print("Look for a URL ending in '.gradio.live' in the console output.")
    demo.launch(share=True, debug=True)
    # --- END EDIT ---
