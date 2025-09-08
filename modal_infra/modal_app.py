import modal
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import io

# Provide constants openly as this is pickled
TRENDYOL_DINOV2_MODEL_NAME = "Trendyol/trendyol-dino-v2-ecommerce-256d"
MODAL_APP_NAME = "trendyol-dino-inference"

# Modal app & container (dependencies)
app = modal.App(MODAL_APP_NAME)
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "timm", "torchvision", "transformers", "Pillow", "opencv-python-headless")
)


# Global function to load/cache the model and processor
def load_model(model_name=TRENDYOL_DINOV2_MODEL_NAME):
    processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    return processor, model


@app.function(image=image, gpu="any", timeout=300)
def run_inference(image_bytes: bytes, model_name=TRENDYOL_DINOV2_MODEL_NAME):
    if not hasattr(run_inference, "processor") or not hasattr(run_inference, "model"):
        run_inference.processor, run_inference.model = load_model(model_name)
        run_inference.model.to("cuda")
        run_inference.model.eval()

    # Convert bytes to PIL Image
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Processor handles conversion & normalization
    inputs = run_inference.processor(images=pil_image, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = run_inference.model(**inputs)
        # Return the full latent representation
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            embeddings = outputs.pooler_output  # [batch_size, hidden_dim]
        else:
            embeddings = outputs.last_hidden_state  # [batch_size, seq_len, hidden_dim]

    return embeddings.cpu().numpy()


