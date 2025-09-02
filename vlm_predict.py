import os
from utils.knowledge import captions
from models.vlm import ClipCaptioner
from time import time

# Example usage
model_path = "train/checkpoints/clipcap_epoch_9.pt"
    
# First initialize the model with desired parameters
model = ClipCaptioner()

# Then load the pretrained weights
model = model.from_pretrained(model_path)

img_path = "test_img.jpg"

# Use the model to generate a caption from the given image
probs, caption, combined_plot, individual_plots = model.generate(image=img_path, combined_alpha=0.4)

# Define an output folder
output_folder = "results/vlm_predict"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Save the combined plot
combined_plot.save(f"{output_folder}/combined_plot.png")

# Save the individual plots
for i, plot in enumerate(individual_plots):
    plot.save(f"{output_folder}/individual_plot_{i}.png")

print(f"Caption: {caption}")    
print(f"All probabilities: {probs}")
    
"""
# For each image in the test folder, generate a caption
data_folder = "/home/g.hauber/TESI/PIPELINE/data/test"
for image_path in os.listdir(data_folder):
    
    # Take the starting time 
    start_time = time()
    
    # Use the model to generate a caption from the given image
    caption = model.generate_caption(
        image_path=os.path.join(data_folder, image_path),
        max_length=77,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        min_length=10,
        prefix_text="A plant disease description: "
    )
    
    # Stop taking time 
    end_time = time()

    # Print info 
    print(f"Generated caption for image {image_path}: {caption}")
    print(f"Time taken: {end_time - start_time} seconds")
    print("--------------------------------")
"""
