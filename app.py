import sys
import subprocess
import platform
import os


def ensure_package(import_name, install_spec):
    try:
        __import__(import_name)
    except ImportError:
        print(f"Installing {install_spec} dynamically...")
        command = [sys.executable, "-m", "pip", "install", "--no-cache-dir"]
        if install_spec.startswith("git+https://github.com/facebookresearch/detectron2.git"):
            command.append("--no-build-isolation")
        command.append(install_spec)
        subprocess.check_call(command)

DISABLE_RUNTIME_BOOTSTRAP = os.getenv("DISABLE_RUNTIME_BOOTSTRAP", "0") == "1"
IS_WINDOWS = platform.system().lower().startswith("win")

# 1. HACK: Dynamically install the runtime stack if Hugging Face starts without it.
if not DISABLE_RUNTIME_BOOTSTRAP:
    ensure_package("torch", "torch==2.11.0")
    ensure_package("torchvision", "torchvision==0.26.0")

    if not IS_WINDOWS:
        ensure_package("detectron2", "git+https://github.com/facebookresearch/detectron2.git")

# 2. Now it is safe to import everything else
import gradio as gr
import cv2, torch
import numpy as np
from PIL import Image
import spaces

# Define the classes mapped during your training
CLASS_NAMES = ["building", "boundary_or_road"]
PREDICTOR = None
DETECTRON2_ERROR = None


def get_detectron2_components():
    global DETECTRON2_ERROR

    try:
        from detectron2.config import get_cfg
        from detectron2 import model_zoo
        from detectron2.engine import DefaultPredictor
        from detectron2.utils.visualizer import Visualizer, ColorMode
        from detectron2.data import MetadataCatalog
        return get_cfg, model_zoo, DefaultPredictor, Visualizer, ColorMode, MetadataCatalog
    except ImportError as exc:
        DETECTRON2_ERROR = exc
        return None


def build_model():
    detectron2_components = get_detectron2_components()
    if detectron2_components is None:
        raise DETECTRON2_ERROR

    get_cfg, model_zoo, DefaultPredictor, _, _, _ = detectron2_components
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CLASS_NAMES)
    cfg.MODEL.WEIGHTS = "model_final.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.40
    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"
    return DefaultPredictor(cfg)


def get_predictor():
    global PREDICTOR
    if PREDICTOR is None:
        PREDICTOR = build_model()
    return PREDICTOR


detectron2_components = get_detectron2_components()
if detectron2_components is not None:
    _, _, _, _, _, MetadataCatalog = detectron2_components
    MetadataCatalog.get("app_catalog").set(thing_classes=CLASS_NAMES)

def resize_for_inference(image_rgb, max_side=1280):
    height, width = image_rgb.shape[:2]
    longest_side = max(height, width)

    if longest_side <= max_side:
        return image_rgb

    scale = max_side / float(longest_side)
    new_width = int(width * scale)
    new_height = int(height * scale)
    return cv2.resize(image_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)


def simple_cpu_fallback(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    preview = image_rgb.copy()
    kept = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 1000:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)
        kept += 1

    message = f"CPU fallback preview generated {kept} region(s)."
    return preview, message


def predict_cadastral_core(input_image):
    if input_image is None:
        return None, "Upload a valid drone survey image."
    
    # Gradio can hand us either a numpy array or a PIL image depending on the component/runtime.
    if isinstance(input_image, np.ndarray):
        image_rgb = input_image
    else:
        image_rgb = np.array(input_image)

    if image_rgb.ndim == 2:
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
    elif image_rgb.shape[-1] == 4:
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGBA2RGB)

    image_rgb = resize_for_inference(image_rgb)

    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    detectron2_components = get_detectron2_components()
    if detectron2_components is None:
        if IS_WINDOWS and not DISABLE_RUNTIME_BOOTSTRAP:
            return simple_cpu_fallback(image_rgb)
        return simple_cpu_fallback(image_rgb)

    _, _, _, Visualizer, ColorMode, MetadataCatalog = detectron2_components

    try:
        predictor = get_predictor()
        outputs = predictor(img_bgr)
        instances = outputs.get("instances")

        if instances is None or len(instances) == 0:
            return image_rgb, "No structures detected in this image."

        v = Visualizer(
            image_rgb,
            metadata=MetadataCatalog.get("app_catalog"),
            scale=1.0,
            instance_mode=ColorMode.IMAGE,
        )
        out = v.draw_instance_predictions(instances.to("cpu"))
        result_rgb = out.get_image()

        count = len(instances)
        status_text = f"Successfully segmented {count} cadastral structures with polygon boundaries."
        return result_rgb, status_text
    except Exception as exc:
        if IS_WINDOWS:
            return simple_cpu_fallback(image_rgb)
        raise exc


@spaces.GPU(duration=10)
def predict_cadastral_gpu(input_image):
    return predict_cadastral_core(input_image)


def predict_cadastral_cpu(input_image):
    return predict_cadastral_core(input_image)


def predict_cadastral(input_image, mode):
    if mode == "GPU":
        return predict_cadastral_gpu(input_image)
    return predict_cadastral_cpu(input_image)

# Create UI
with gr.Blocks(title="AI Cadastral Mapping") as demo:
    gr.Markdown("# High-Precision Cadastral Boundary Extraction")
    gr.Markdown("Instance Segmentation Pipeline powered by **Mask R-CNN (Detectron2)**.")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="numpy", label="Upload Raw Drone Survey (JPG/PNG)")
            mode = gr.Radio(["CPU", "GPU"], value="CPU", label="Inference Mode")
            submit_btn = gr.Button("Extract Boundaries", variant="primary")
        
        with gr.Column():
            output_img = gr.Image(type="numpy", label="Segmented Cadastral Polygons")
            output_info = gr.Textbox(label="Detection Summary")
            
    submit_btn.click(
        fn=predict_cadastral,
        inputs=[input_img, mode],
        outputs=[output_img, output_info]
    )

if __name__ == "__main__":
    demo.launch(server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"), server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")))