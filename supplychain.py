import gradio as gr
import os
from PIL import Image
from models.vlm import ClipCaptioner # Ensure this import path is correct for your project structure
import traceback # For detailed error logging

# --- Configuration ---
# Consider making this configurable via environment variables or arguments
MODEL_PATH = "train/checkpoints/clipcap_epoch_9.pt"
TEMP_INPUT_FILENAME = "temp_gradio_input.jpg" # Temporary file for processing

# --- Model Loading ---
model = None
model_load_error = None
print("Attempting to load VLM model...")
try:
    if os.path.exists(MODEL_PATH):
        model = ClipCaptioner()
        model = model.from_pretrained(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}.")
    else:
        model_load_error = f"Model file not found at {MODEL_PATH}. Please check the path."
        print(model_load_error)
except Exception as e:
    model_load_error = f"Error loading model: {e}\n{traceback.format_exc()}"
    print(model_load_error)
    model = None # Ensure model is None if loading failed

# --- Processing Function ---
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
        image_input_pil (PIL.Image.Image): Input image from Gradio.
        combined_alpha_val (float): Alpha value for combined plot.
        temperature_val (float): Sampling temperature for generation.
        prefix_text_val (str): Prefix text for caption generation.
        normalize_alpha_val (bool): Flag for normalizing alpha scaling.


    Returns:
        tuple: Contains combined plot, list of individual plots, top probabilities text,
               and caption text. Returns error indicators if processing fails.
    """
    if model is None:
        error_msg = f"Model could not be loaded. Please check logs. Error details: {model_load_error}"
        return None, [], error_msg, error_msg
    if image_input_pil is None:
         return None, [], "Please upload an image and click 'Generate'.", ""

    img_path = TEMP_INPUT_FILENAME
    try:
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

        # --- Format Outputs ---

        # 1. Top 5 Probabilities
        top_5_text = "Top 5 Predicted Captions:\n"
        if probs and isinstance(probs, dict):
            # Sort probabilities in descending order
            sorted_probs = sorted(probs.items(), key=lambda item: item[1], reverse=True)
            top_5 = sorted_probs[:5]
            if top_5:
                 top_5_text += "\n".join([f"{i+1}. {cap} (Prob: {prob:.4f})" for i, (cap, prob) in enumerate(top_5)])
            else:
                 top_5_text += "No probabilities were returned by the model."
        else:
            top_5_text += f"Probabilities not available or in unexpected format: {type(probs)}"

        # 2. Individual Plots Gallery
        # Ensure individual_plots is a list and filter for valid PIL Images
        individual_plots_list = []
        if individual_plots and isinstance(individual_plots, list):
            valid_plots_found = False
            for i, plot in enumerate(individual_plots):
                if isinstance(plot, Image.Image):
                    individual_plots_list.append(plot)
                    valid_plots_found = True
                else:
                    print(f"Warning: Item at index {i} in individual_plots is not a PIL Image: {type(plot)}")
            if not valid_plots_found:
                 print("Warning: No valid PIL Images found in individual_plots.")
        elif individual_plots:
             print(f"Warning: individual_plots is not a list: {type(individual_plots)}")
        else:
             print("Warning: No individual plots were generated or returned.")


        # 3. Combined Plot
        # Ensure combined_plot is a PIL Image
        if not isinstance(combined_plot, Image.Image):
            print(f"Warning: combined_plot is not a PIL Image: {type(combined_plot)}")
            combined_plot = None # Set to None if not an image

        # 4. Caption
        if not isinstance(caption, str):
            print(f"Warning: Generated caption is not a string: {type(caption)}")
            caption = str(caption) # Attempt to convert to string

        print(f"Generated Caption: {caption}")
        print(f"Formatted Top 5 Probs: {top_5_text}")
        print(f"Number of individual plots returned for gallery: {len(individual_plots_list)}")

        # Return the list of individual plots for the gallery
        return combined_plot, individual_plots_list, top_5_text, caption


    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error during prediction: {e}\n{error_details}")
        # Return empty list for the gallery on error
        return None, [], f"Error during processing: {e}", f"Error: {e}"
    finally:
        # Clean up the temporary image file
        if os.path.exists(img_path):
            try:
                os.remove(img_path)
                print(f"Temporary file {img_path} removed.")
            except OSError as e:
                print(f"Error removing temporary file {img_path}: {e}")


# --- Gradio Interface Definition ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# VLM Prediction Visualization")
    gr.Markdown("Upload an image and adjust parameters to generate a caption and visualize attention maps.")

    with gr.Row():
        # Input Column
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Input Image")

            # --- Add New Input Controls ---
            with gr.Accordion("Generation Parameters", open=False): # Use Accordion for tidiness
                input_prefix_text = gr.Textbox(
                    label="Caption Prefix",
                    value="A plant disease description: ",
                    info="Text to prepend to the generated caption."
                )
                input_combined_alpha = gr.Slider(
                    minimum=0.0, maximum=1.0, step=0.05, value=0.4,
                    label="Combined Plot Alpha",
                    info="Transparency of attention overlay on the combined plot."
                )
                input_temperature = gr.Slider(
                    minimum=0.0, maximum=1.0, step=0.05, value=0.7,
                    label="Temperature",
                    info="Controls randomness in generation (higher is more random)."
                )
                input_normalize_alpha = gr.Checkbox(
                    label="Normalize Alpha Scaling",
                    value=True, # Default value
                    info="Whether to normalize attention map scaling."
                )
            # --- End New Input Controls ---

            generate_button = gr.Button("Generate Caption & Visualize", variant="primary", interactive=model is not None)

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

# --- Launch the App ---
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