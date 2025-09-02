
import io
import cv2
import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from PIL import Image

# Helper function to plot the Grad-CAM map overlay
def individual_map(original_img, cam_map, caption, probability=None, colormap="jet"):
    """
    Plots the Grad-CAM map overlaid on the original image and saves it to disk.

    Args:
        original_img (PIL.Image): The original input image.
        cam_map (np.ndarray): The 2D Grad-CAM heatmap (normalized 0-1).
        output_path (str): Path to save the visualization.
        caption (str): The target caption used for Grad-CAM.
        probability (float, optional): The probability score for the caption. Defaults to None.
        colormap (str): Matplotlib colormap name.
    """

    # Set the matplotlib backend to Agg (non-interactive)
    matplotlib.use('Agg')

    # Apply colormap to the CAM map and convert to RGB uint8
    heatmap = cm.get_cmap(colormap)(cam_map)[:, :, :3] 
    heatmap = (heatmap * 255).astype(np.uint8) 

    # Resize heatmap to original image size using PIL for potentially better quality
    # and use bilinear interpolation 
    heatmap_img = Image.fromarray(heatmap)
    heatmap_resized = heatmap_img.resize(original_img.size, Image.BILINEAR) 

    # Overlay heatmap onto original image
    overlay_img = Image.blend(original_img, heatmap_resized, alpha=0.5)

    # Create figure to save
    fig, axs = plt.subplots(1, 2, figsize=(12, 5)) # Increase figure size slightly

    # Original Image
    axs[0].imshow(original_img)
    axs[0].set_title("Original Image")
    axs[0].axis('off')

    # Grad-CAM Overlay
    axs[1].imshow(overlay_img)
    title = f"Grad-CAM: '{caption[:50]}...'"
 
    # If probability is not None, we add it to the title
    if probability is not None:

        # Add the probability to the title
        title = f"Grad-CAM (Prob: {probability:.3f}): '{caption[:40]}...'"

    # Set the title
    axs[1].set_title(title)

    # Turn off the axis
    axs[1].axis('off')

    # Save the figure
    plt.tight_layout()
    
    # Convert plot to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_image = Image.open(buf)

    # Close the figure to free memory
    plt.close(fig)

    # Return the image
    return plot_image

# Overlays multiple Grad-CAM maps onto a single image with distinct colors and a legend.
# The opacity (alpha) of each overlay region is scaled by the provided 'probabilities_for_alpha'
# (either original or normalized), multiplied by the 'alpha' parameter (max alpha).
# The legend displays the ORIGINAL probabilities.
def combined_map(original_img, 
                              captions, 
                              cam_maps,
                              probabilities_for_alpha, 
                              original_probabilities, 
                              k, 
                              threshold, 
                              alpha,
                              normalize_alpha_scaling, 
                              colormap_name="jet"):
    
    # Brief explanation of the function parameters since it's a bit complex:
    """
    Args:
        original_img (PIL.Image): The original input image.
        captions (list[str]): List of top K captions.
        cam_maps (list[np.ndarray]): List of corresponding K Grad-CAM heatmaps (normalized 0-1, 2D).
        probabilities_for_alpha (list[float] | None): List of corresponding K NORMALIZED probabilities
                                                     (sum=1) for alpha scaling if normalize_alpha_scaling is True.
                                                     Otherwise, this is None and ignored.
        original_probabilities (list[float]): List of corresponding K ORIGINAL probabilities for legend display.
        output_path (str): Path to save the combined visualization.
        k (int): Number of top captions/maps.
        colormap_name (str): Matplotlib colormap name for distinct colors (e.g., 'tab10', 'Set1').
        threshold (float): Minimum CAM value (0-1) to consider for overlay.
        alpha (float): Maximum blending factor for the heatmap overlay (0-1).
                       Effective alpha = probability_for_alpha * alpha.
        normalize_alpha_scaling (bool): If True, scale alpha by normalized probabilities.
                                        If False, use constant alpha.
    """

    # Set the matplotlib backend to Agg (non-interactive)
    matplotlib.use('Agg')

    # Convert the original image to a numpy array
    img_np = np.array(original_img)

    # Get the height and width of the image
    height, width, _ = img_np.shape

    # Get K distinct colors from the chosen colormap
    colormap = cm.get_cmap(colormap_name)
    colors = [colormap(i / k)[:3] for i in range(k)] # Get RGB, discard alpha from colormap

    # Prepare overlay canvas (float for blending)
    overlay = np.zeros((height, width, 3), dtype=np.float32)

    # Mask to track which pixels have been colored
    colored_mask = np.zeros((height, width), dtype=bool)

    # Resize CAM maps and stack them
    resized_cams = []
    for cam_map in cam_maps:

        # Resize using OpenCV (ensure cam_map is float32)
        resized_cam = cv2.resize(cam_map.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
        
        # Append the resized CAM map to the list
        resized_cams.append(resized_cam)

    # Stack the resized CAM maps
    stacked_cams = np.stack(resized_cams, axis=0)

    # Apply threshold
    thresholded_cams = np.where(stacked_cams > threshold, stacked_cams, 0)

    # Find which map has the max value (above threshold) for each pixel
    # Shape: (height, width)
    max_indices = np.argmax(thresholded_cams, axis=0) 
    max_values = np.max(thresholded_cams, axis=0)     

    # Create the overlay based on max index and threshold
    # Pixels where at least one CAM was above threshold
    valid_mask = max_values > 0 

    # Iterate over the top K captions
    for i in range(k):

        # Find pixels where the i-th map was the maximum and above threshold
        map_mask = valid_mask & (max_indices == i)

        # Assign the color for the i-th caption to these pixels
        overlay[map_mask] = colors[i]

    # Blend the overlay with the original image
    img_float = img_np.astype(np.float32) / 255.0

    # Create a copy of the original image
    combined_img = img_float.copy()

    # Apply alpha blending based on the flag
    for i in range(k):
        map_mask = valid_mask & (max_indices == i)

        # Calculate alpha based on the flag
        if normalize_alpha_scaling:

            # Scale by normalized probability (ensure probabilities_for_alpha is provided)
            if probabilities_for_alpha is None:
                 
                 # This case should ideally not happen if called correctly from visualize_gradcam.
                 # We fallback to constant alpha.
                 current_alpha = alpha 

            # Otherwise, we scale by the normalized probability
            else:
                current_alpha = np.clip(probabilities_for_alpha[i] * alpha, 0.0, 1.0)

        # Otherwise, we use a constant alpha
        else:
            
            # Use constant alpha
            current_alpha = alpha

        # Apply blending
        combined_img[map_mask] = current_alpha * overlay[map_mask] + (1 - current_alpha) * img_float[map_mask]

    # Pixels where no CAM was above threshold remain unchanged (already handled by combined_img = img_float.copy())
    # Create Plot with Legend
    fig, ax = plt.subplots(1, 1, figsize=(10, 6)) # Adjust figsize as needed

    # Clip to ensure valid range [0, 1] after blending
    ax.imshow(np.clip(combined_img, 0, 1))
    ax.axis('off')

    # Create legend handles using ORIGINAL probabilities
    legend_handles = []

    # Iterate over the top K captions
    for i in range(k):

        # Get the caption
        label = captions[i]

        # Get the original probability
        prob = original_probabilities[i]

        # If the caption is too long, we truncate it
        if len(label) > 35:
            label = label[:32] + "..."

        # Create a patch for the legend
        patch = mpatches.Patch(color=colors[i], label=f"Rank {i+1} ({prob:.3f}): {label}")

        # Add the patch to the legend handles
        legend_handles.append(patch)

    # Add legend outside the plot area
    ax.legend(handles=legend_handles, bbox_to_anchor=(0.5, -0.05), loc='upper center', ncol=1, borderaxespad=0.)

    # Update title to reflect the alpha method used
    scaling_method = "NormProb-Scaled" if normalize_alpha_scaling else "Constant-Alpha"
    plt.suptitle(f"Top {k} Grad-CAM Overlay (Thresh: {threshold}, Max Alpha: {alpha}, {scaling_method})", fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Convert plot to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_image = Image.open(buf)

    # Close the figure to free memory
    plt.close(fig)

    # Return the image
    return plot_image