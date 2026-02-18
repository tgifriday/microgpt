import torch

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Move model to GPU
model.to(device)

# Move input data to GPU
input_data = input_data.to(device)

# Perform inference
output = model(input_data)

