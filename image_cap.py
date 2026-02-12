import requests
from PIL import Image

from transformers import AutoProcessor, BlipForConditionalGeneration


processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


img_path = "/home/aswanth.cp@knackforge.com/Public/ML/Projects/GiveMeaningfulNameToYourPhotos/Media.jpeg"

image = Image.open(img_path).convert('RGB')

text = "a image of "

inputs = processor(image, text, return_tensors="pt")

out = model.generate(**inputs, max_length=150)

caption = processor.decode(out[0], skip_special_tokens=True)
print(caption)