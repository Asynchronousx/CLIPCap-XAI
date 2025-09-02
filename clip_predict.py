from models.wrapper import CLIPW
from utils.knowledge import captions

# Instantiate the CLIPWrapper class
clip_wrapper = CLIPW(captions=captions["fruits"])

# Load the image path and the captions 
image_path = "test_img.jpg"

# Predict what's in the image: probs, image_features, text_features are returned
# We only need the probabilities since we didn't specify return_img_emb or return_text_emb.
# So we just get the first element of the tuple.
# Also, we don't need to specify the captions since we already have them saved.
probs = clip_wrapper.predict(image_path)[0]

# Extract the top 5 captions and combine them with their probabilities 
# from the model output
top_predictions = clip_wrapper.top_k_predictions(probs, k=5)

# print the top predictions
print(top_predictions)

# Synthesize a caption from the model output
caption = clip_wrapper.synthesize_caption(probs)

# print the synthesized caption
print(caption)