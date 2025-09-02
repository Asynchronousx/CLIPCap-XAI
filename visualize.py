from models.clip import CLIPW
from utils.knowledge import captions
import os

# TEST
from models.vlm import ClipCaptioner
clipcap = ClipCaptioner()
#clipcap.from_pretrained("train/checkpoints/clipcap_epoch_9.pt")

# Create a CLIPW object
clip_wrapper = CLIPW(captions=captions["fruits"])

# Define the image file and output folder
image_file = "test_img.jpg"
output_folder_path = "results/results_path"
output_folder_cv2 = "results/results_cv2"
output_folder_pil = "results/results_pil"

# TEST: open the image with various methods
from PIL import Image   

# With PIL
original_img_pil = Image.open(image_file)

# With openCV
import cv2
original_img_cv2 = cv2.imread(image_file)


# Get the visual explanation
out = clipcap.get_visual_explanation(image=image_file)

# Get the visual explanation with PIL
out_pil = clipcap.get_visual_explanation(image=original_img_pil)

# Get the visual explanation with openCV
out_cv2 = clipcap.get_visual_explanation(image=original_img_cv2)


# Unpack the output
probs,image_features, combined_plot_image, individual_plot_images = out

# Unpack the output with PIL
probs_pil,image_features_pil, combined_plot_image_pil, individual_plot_images_pil = out_pil

# Unpack the output with openCV
probs_cv2,image_features_cv2, combined_plot_image_cv2, individual_plot_images_cv2 = out_cv2

# Print probs
print(probs)

# Print probs with PIL
print(probs_pil)

# Print probs with openCV
print(probs_cv2)


# Create the specified output directory if it doesn't already exist
os.makedirs(output_folder_path, exist_ok=True)
os.makedirs(output_folder_cv2, exist_ok=True)
os.makedirs(output_folder_pil, exist_ok=True)

# save combined plot image
combined_plot_image.save(f"{output_folder_path}/combined_plot.png")
combined_plot_image_pil.save(f"{output_folder_pil}/combined_plot.png")
combined_plot_image_cv2.save(f"{output_folder_cv2}/combined_plot.png")

# save individual plot images
for i, image in enumerate(individual_plot_images):
    image.save(f"{output_folder_path}/individual_plot_{i}.png")
for i, image in enumerate(individual_plot_images_pil):
    image.save(f"{output_folder_pil}/individual_plot_{i}.png")
for i, image in enumerate(individual_plot_images_cv2):
    image.save(f"{output_folder_cv2}/individual_plot_{i}.png")

"""
# Image path 
image_file = "test_img.jpg"
output_folder = "results/single_gradcam_test"

# We call visualize_gradcam only with image_path and output_dir. 
# This will produce a single Grad-CAM map for the most confident caption.
clip_wrapper.visualize_gradcam(
    image_path=image_file,
    output_dir=output_folder,
)

# Declaring another output folder
output_folder = "results/gradcam_top3"

# We call visualize_gradcam with k=3 and plot_individual=True.
# This will produce 3 individual Grad-CAM maps for the top 3 captions, 
# alongside with the combined overlay plot.
# Note that we don't set normalize_alpha_scaling to True, so the alpha values are not normalized
# by the top 3 probabilities. (hence, all alpha values are the same).
clip_wrapper.visualize_gradcam(
    image_path=image_file,
    output_dir=output_folder,
    k=3,
    plot_individual=True # Default, saves individual maps
)

# Declaring another output folder
output_folder = "results/gradcam_top5_combined"

# We call visualize_gradcam with k=5 and plot_individual=False.
# This will produce a single combined Grad-CAM map for the top 5 captions, 
# because we set plot_individual to False. We also set a custom name and adjust 
# the threshold and transparency of the overlay.
# Also we set normalize_alpha_scaling to True, so that the alpha values are normalized
# by the top 5 probabilities. (most confident captions have higher alpha values)
clip_wrapper.visualize_gradcam(
    image_path=image_file,
    output_dir=output_folder,
    k=5,
    plot_individual=False, # Only save the combined plot
    combined_output_filename="top5_overlay.png", # Custom name for combined plot
    combined_threshold=0.4, # Adjust threshold for overlay
    combined_alpha=1.0,      # Adjust transparency
    normalize_alpha_scaling=True # Normalize alpha scaling
)
"""



"""
# Folder evaluation
# For each image in the folder 
output_folder = "results/test_folder"

import os
for image_path in os.listdir("data/test"):

    # Print the starting message
    print(f"Starting gradcam for {image_path}")

    # Create a subfolder for the attention rollout images
    os.makedirs(f"data/test/gradcam", exist_ok=True)

    # Get the image path
    image_path = f"data/test/{image_path}"

    # Get the image name
    image_name = image_path.split("/")[-1]

    # check if the image is not a directory
    if not os.path.isdir(image_path):

        # Get the gradcam image 
        clip_wrapper.visualize_gradcam(
            image_path=image_path,
            output_dir=output_folder,
            k=5,
            plot_individual=True, # Only save the combined plot
            combined_output_filename=f"{image_name}_top5_overlay.png", # Custom name for combined plot
            combined_threshold=0.0, # Adjust threshold for overlay
            combined_alpha=1.0,      # Adjust transparency
            normalize_alpha_scaling=True # Normalize alpha scaling
        )

        # Print the completion message
        print(f"Saved visual cues for {image_name}")
"""
