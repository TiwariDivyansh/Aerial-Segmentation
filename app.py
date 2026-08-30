import sys
import subprocess

# 1. HACK: Dynamically install Detectron2 on startup to bypass build isolation
try:
    import detectron2
except ImportError:
    print("Installing detectron2 dynamically...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/facebookresearch/detectron2.git"])

# 2. Now it is safe to import everything else
import gradio as gr
import cv2, os, torch
import numpy as np
from PIL import Image
import spaces

import detectron2
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

# Define the classes mapped during your training
CLASS_NAMES = ["building", "boundary_or_road"]

def build_model():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CLASS_NAMES)
    cfg.MODEL.WEIGHTS = "model_final.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.40
    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"
    return DefaultPredictor(cfg)

# Initialize predictor globally
predictor = build_model()

MetadataCatalog.get("app_catalog").set(thing_classes=CLASS_NAMES)

@spaces.GPU(duration=30)
def predict_cadastral(input_image):
    if input_image is None:
        return None, "Upload a valid drone survey image."
    
    # Gradio passes images in RGB numpy format
    img_bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    
    # Run Mask R-CNN inference
    outputs = predictor(img_bgr)
    
    # Draw colored instance segmentations
    v = Visualizer(
        input_image, 
        metadata=MetadataCatalog.get("app_catalog"), 
        scale=1.0, 
        instance_mode=ColorMode.COLOR
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    result_rgb = out.get_image()
    
    count = len(outputs["instances"])
    status_text = f"Successfully segmented {count} cadastral structures with polygon boundaries."
    
    return result_rgb, status_text

# Create UI
with gr.Blocks(title="AI Cadastral Mapping") as demo:
    gr.Markdown("# High-Precision Cadastral Boundary Extraction")
    gr.Markdown("Instance Segmentation Pipeline powered by **Mask R-CNN (Detectron2)**.")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="numpy", label="Upload Raw Drone Survey (JPG/PNG)")
            submit_btn = gr.Button("Extract Boundaries", variant="primary")
        
        with gr.Column():
            output_img = gr.Image(type="numpy", label="Segmented Cadastral Polygons")
            output_info = gr.Textbox(label="Detection Summary")
            
    submit_btn.click(
        fn=predict_cadastral,
        inputs=[input_img],
        outputs=[output_img, output_info]
    )

if __name__ == "__main__":
    demo.launch()