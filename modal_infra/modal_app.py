import modal
import torch
from transformers import AutoImageProcessor, AutoModel
import io
from PIL import Image
from modal_infra.constants import *


class DinoV2Inference:
    """Encapsulates Modal deployment for Trendyol DINO v2."""

    def __init__(self, model_name=TRENDYOL_DINOV2_MODEL_NAME, app_name=MODAL_APP_NAME):
        self.model_name = model_name
        self.app_name = app_name
        self.app = modal.App(app_name)

        # Modal container (dependencies)
        self.image = (
            modal.Image.debian_slim()
            .pip_install("torch", "transformers", "Pillow")
        )

        # Attach function dynamically to app
        self._register_modal_function()

    def _register_modal_function(self):
        model_name = self.model_name

        @self.app.function(image=self.image, gpu="any", timeout=300)
        def run_inference(image_bytes: bytes):
            # Cache model & processor directly on the function object
            if not hasattr(run_inference, "model"):
                run_inference.processor = AutoImageProcessor.from_pretrained(
                    model_name, trust_remote_code=True
                )
                run_inference.model = AutoModel.from_pretrained(
                    model_name, trust_remote_code=True
                )

            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            inputs = run_inference.processor(images=img, return_tensors="pt")

            with torch.no_grad():
                outputs = run_inference.model(**inputs)
                if hasattr(outputs, "last_hidden_state"):
                    embeddings = outputs.last_hidden_state.mean(dim=1)
                elif hasattr(outputs, "pooler_output"):
                    embeddings = outputs.pooler_output
                else:
                    raise ValueError("Unexpected model outputs")

            return embeddings.cpu().numpy().tolist()

        self.run_inference = run_inference

