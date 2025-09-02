from models.vlm import ClipCaptioner

# Your directory path: 
# - Root directory: where the dataset is located
# - Processed directory: where the processed dataset is gonna be saved
# - Output directory: where the checkpoints are gonna be saved
root_dir = "/home/g.hauber/TESI/PIPELINE/data/disease_dataset/Rotten"
processed_dir = "/home/g.hauber/TESI/PIPELINE/train/processed_dataset"
output_dir = "/home/g.hauber/TESI/PIPELINE/train/checkpoints"

# Initialize the VLM model 
model = ClipCaptioner(
    prefix_length=10,
    clip_length=512,
    hidden_size=768,
    num_layers=8,
    num_heads=8
)

# Preprocess the dataset if needed
model.preprocess_dataset(
    root_dir=root_dir,
    output_dir=processed_dir,
    batch_size=32
)

# train the model and save the checkpoints 
model.train_model(
    dataset_path=processed_dir,
    output_dir=output_dir,
    epochs=10,
    batch_size=32
)

