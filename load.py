import torch
from model import SimpleMultiHeadSigmoid, MultiHeadSigmoid
from config import backbone_model_name


def load_model(check_point: str):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(check_point, map_location=torch.device(DEVICE))
    model = SimpleMultiHeadSigmoid(
        backbone_model_name,
        checkpoint["num_aspects"],
        checkpoint["num_sentiments"],
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(DEVICE)
    return model
