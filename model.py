import torch


# Load model
def load_model(model_file, force_device=None):

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    model = torch.load(model_file, map_location=torch.device(device))

    return model
