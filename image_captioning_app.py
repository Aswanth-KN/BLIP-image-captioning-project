from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration
import numpy as np
import gradio as gr

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")




def caption_image(image):
    image = Image.fromarray(image).convert('RGB')
    text = "a image of "
    inputs = processor(image, text, return_tensors="pt")
    out = model.generate(**inputs, max_length=50)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption





iface = gr.Interface(
    fn=caption_image, 
    inputs=gr.Image(), 
    outputs="text",
    title="Image Captioning",
    description="This is a simple web app for generating captions for images using a trained model."
)

iface.launch()