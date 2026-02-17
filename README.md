# Give Meaningful Name To Your Photos 📸

An AI-powered image captioning system that generates descriptive captions for images using Salesforce's BLIP model.

## 📋 Overview

This project uses state-of-the-art vision-language models to automatically generate meaningful, descriptive captions for images. Whether you want to organize your photo library, make images searchable, or add accessibility features, this tool makes it easy.

## ✨ Features

- **Automatic Caption Generation**: AI-powered descriptions for any image
- **Web Scraping**: Extract and caption images from websites
- **Local Image Processing**: Batch process images from your computer
- **Interactive Web UI**: Upload and caption images through a browser
- **Multiple Input Methods**: Web, local files, or URLs
- **Text File Export**: Save captions for later use

## 🗂️ Project Structure

```
GiveMeaningfulNameToYourPhotos/
├── image_captioning_app.py           # Gradio web interface (single image)
├── automatic_image_captioning.py     # Web scraping and bulk captioning
├── automatic_image_from_local.py     # Local file processing
├── image_cap.py                      # Core captioning utility
├── captions.txt                      # Output file with generated captions
└── README.md                         # This file
```

## 📦 Installation

### 1. Set Up Virtual Environment

```bash
# Navigate to project root
cd /home/aswanth.cp@knackforge.com/Public/ML/Projects

# Activate virtual environment
source env/bin/activate
```

### 2. Install Dependencies

```bash
cd GiveMeaningfulNameToYourPhotos
pip install transformers torch gradio pillow beautifulsoup4 requests numpy
```

### Required Packages

- `transformers` - Hugging Face transformers library
- `torch` - PyTorch deep learning framework
- `gradio` - Web UI framework
- `pillow` (PIL) - Image processing library
- `beautifulsoup4` - HTML/XML parsing for web scraping
- `requests` - HTTP library for downloading images
- `numpy` - Numerical computing library

## 🚀 Usage

### Option 1: Interactive Web Interface

Perfect for captioning individual images with a user-friendly interface.

```bash
python3 image_captioning_app.py
```

**Access:** `http://localhost:7860`

**How to Use:**
1. Open the web interface in your browser
2. Click or drag to upload an image
3. Click "Submit"
4. View the generated caption

**Supported Formats:**
- JPG/JPEG
- PNG
- WebP
- BMP
- GIF (first frame)

### Option 2: Web Scraping (Bulk Processing)

Automatically extract and caption all images from a webpage.

```bash
python3 automatic_image_captioning.py
```

**What It Does:**
1. Scrapes images from specified URL (default: Wikipedia IBM page)
2. Filters out small images, SVGs, and icons
3. Generates captions for each valid image
4. Saves results to `captions.txt`

**Output Format:**
```
https://example.com/image1.jpg: the image of a laptop computer on a desk
https://example.com/image2.jpg: the image of a person sitting in front of a building
https://example.com/image3.jpg: the image of a smartphone with colorful screen
```

**Customize the URL:**

Edit `automatic_image_captioning.py`:
```python
url = "https://your-target-website.com"  # Change this line
```

### Option 3: Local File Processing

Process images from your local computer.

```bash
python3 automatic_image_from_local.py
```

**Setup:**
1. Place images in a designated folder
2. Update the script with your folder path
3. Run the script
4. Captions saved to `captions.txt`

### Option 4: Core Utility

Use as a module in your own scripts.

```python
from image_cap import generate_caption

caption = generate_caption("path/to/image.jpg")
print(caption)
```

## 🔧 File Details

### image_captioning_app.py

**Purpose:** Interactive web interface for single image captioning  
**Best For:** User-friendly image captioning, demos

**Features:**
- Drag-and-drop image upload
- Real-time caption generation
- Clean, modern interface
- No file system access needed

**Code Highlights:**
```python
def caption_image(image):
    # Converts uploaded image to RGB
    # Processes through BLIP model
    # Returns descriptive caption
```

### automatic_image_captioning.py

**Purpose:** Web scraping and bulk image captioning  
**Best For:** Processing images from websites, research, data collection

**Features:**
- Scrapes images from any webpage
- Filters out inappropriate images (SVGs, tiny icons)
- Handles relative and absolute URLs
- Robust error handling
- Progress tracking

**Smart Filtering:**
- Skips images smaller than 200px²
- Ignores SVG files
- Handles various URL formats
- Converts all images to RGB

**Example Usage:**
```python
url = "https://en.wikipedia.org/wiki/Python_(programming_language)"
# Modify the URL variable in the script
```

### automatic_image_from_local.py

**Purpose:** Process local images in batch  
**Best For:** Organizing photo libraries, bulk processing

**Features:**
- Batch processing of local files
- Recursive folder scanning
- Progress indicators
- Export to text file

### image_cap.py

**Purpose:** Core captioning utility  
**Best For:** Integration into other projects

**Functions:**
- `load_model()` - Initialize BLIP model
- `generate_caption(image_path)` - Caption single image
- `batch_process(folder_path)` - Process multiple images

## 🧠 How It Works

### Architecture

```
Input Image → Preprocessing → Vision Encoder → Language Model → Caption
                                     ↓              ↓
                              Image Features   Text Generation
```

### Workflow

1. **Model Loading**
   - Downloads BLIP-image-captioning-base (~1 GB) on first run
   - Loads vision processor and language model

2. **Image Processing**
   - Loads image using PIL
   - Converts to RGB format
   - Resizes if needed
   - Normalizes pixel values

3. **Feature Extraction**
   - Vision transformer encodes image
   - Extracts visual features
   - Creates feature embeddings

4. **Caption Generation**
   - Language model processes features
   - Generates descriptive text
   - Applies beam search or sampling
   - Returns best caption

5. **Post-processing**
   - Removes special tokens
   - Formats output text
   - Returns final caption

### Model Details

**BLIP (Bootstrapping Language-Image Pre-training)**
- **Developer:** Salesforce Research
- **Type:** Vision-language transformer
- **Architecture:** Vision encoder + Language decoder
- **Training:** Large-scale image-text pairs
- **Strengths:** Natural descriptions, object recognition, scene understanding

## ⚙️ Configuration

### Adjust Caption Style

Modify the prompt prefix in any script:

```python
text = "a photograph of"  # More formal
text = "the image shows"  # Descriptive
text = "the image of"     # Default
text = ""                 # Let model decide
```

### Control Caption Length

```python
out = model.generate(**inputs, max_new_tokens=100)  # Longer captions
out = model.generate(**inputs, max_new_tokens=20)   # Shorter captions
```

### Image Size Filtering

In `automatic_image_captioning.py`:

```python
if raw_image.size[0] * raw_image.size[1] < 500:  # More strict
    continue
```

### Custom Web Scraping

```python
# Target specific image classes
img_elements = soup.find_all("img", class_="content-image")

# Target specific containers
container = soup.find("div", id="gallery")
img_elements = container.find_all("img")
```

## 🐛 Troubleshooting

### Image Won't Load

**Check:**
- File format is supported
- File isn't corrupted
- Path is correct
- Sufficient permissions

**Solution:**
```python
try:
    image = Image.open(image_path)
    image = image.convert('RGB')
except Exception as e:
    print(f"Error: {e}")
```

### Web Scraping Issues

**Common Problems:**
- Website blocks automated access
- Images behind authentication
- Dynamically loaded images (JavaScript)

**Solutions:**
```python
# Add user agent
headers = {
    "User-Agent": "Mozilla/5.0 ..."
}

# Increase timeout
response = requests.get(url, timeout=30)

# Handle redirects
response = requests.get(url, allow_redirects=True)
```

### Model Not Downloading

```bash
# Check internet connection
ping huggingface.co

# Manual download
huggingface-cli download Salesforce/blip-image-captioning-base

# Check cache
ls ~/.cache/huggingface/hub/
```

### Poor Quality Captions

**Improvements:**
- Use higher resolution images
- Ensure good lighting in photos
- Try different prompt prefixes
- Use the large BLIP model instead of base

### Memory Issues

```bash
# Clear Python cache
python3 -c "import torch; torch.cuda.empty_cache()"

# Use CPU-only mode
export CUDA_VISIBLE_DEVICES=""
```

## 📊 Performance

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 4 GB | 8 GB+ |
| Storage | 2 GB | 5 GB+ |
| CPU | 2 cores | 4+ cores |
| GPU | Optional | CUDA-capable |

### Processing Times

| Setup | Time per Image |
|-------|----------------|
| CPU (4 cores) | 2-5 seconds |
| GPU (CUDA) | 0.5-1 second |
| High-end GPU | 0.2-0.5 seconds |

### Batch Processing

- **10 images:** ~30 seconds (CPU) / ~5 seconds (GPU)
- **100 images:** ~5 minutes (CPU) / ~1 minute (GPU)
- **1000 images:** ~50 minutes (CPU) / ~10 minutes (GPU)

### Model Size

- Download size: ~1 GB
- Memory footprint: ~2-3 GB RAM
- Cached location: `~/.cache/huggingface/transformers/`

## 🎯 Use Cases

### Photo Organization

```bash
# Caption all photos in a folder
# Rename files based on captions
for img in photos/*; do
    caption=$(python3 image_cap.py "$img")
    # Use caption for naming
done
```

### Accessibility

```html
<!-- Add alt text to images -->
<img src="photo.jpg" alt="the image of a sunset over mountains">
```

### Content Moderation

- Identify inappropriate content
- Filter images by description
- Categorize images automatically

### Search Enhancement

- Make images searchable by content
- Create image databases
- Build reverse image search

### Social Media

- Auto-generate Instagram captions
- Add context to posts
- Improve discoverability

## 🔐 Best Practices

### Web Scraping Ethics

1. **Respect robots.txt**
2. **Add delays between requests**
3. **Don't overload servers**
4. **Check website terms of service**
5. **Attribute sources properly**

```python
import time

for img in images:
    process_image(img)
    time.sleep(1)  # Be respectful
```

### Image Privacy

- Don't caption private/personal images without permission
- Respect copyright
- Handle sensitive content appropriately

### Performance Optimization

```python
# Batch process when possible
images = load_batch(folder)
captions = model.generate_batch(images)  # Faster

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
```

## 🚀 Advanced Usage

### Fine-tuning on Custom Data

```python
from transformers import BlipForConditionalGeneration, AutoProcessor

# Load model
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Fine-tune on your data
# (requires training dataset and training loop)
```

### Integration with Databases

```python
import sqlite3

conn = sqlite3.connect('images.db')
cursor = conn.cursor()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS images (
        id INTEGER PRIMARY KEY,
        filename TEXT,
        caption TEXT,
        timestamp DATETIME
    )
''')

# Store captions
cursor.execute('INSERT INTO images VALUES (?, ?, ?, ?)', 
               (None, filename, caption, datetime.now()))
```

### REST API Wrapper

```python
from flask import Flask, request, jsonify
from PIL import Image
import io

app = Flask(__name__)

@app.route('/caption', methods=['POST'])
def caption():
    file = request.files['image']
    image = Image.open(io.BytesIO(file.read()))
    caption = generate_caption(image)
    return jsonify({'caption': caption})
```

## 📚 Resources

- [BLIP Paper](https://arxiv.org/abs/2201.12086)
- [Hugging Face Model Card](https://huggingface.co/Salesforce/blip-image-captioning-base)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)
- [PIL Documentation](https://pillow.readthedocs.io/)
- [BeautifulSoup Documentation](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)

## 🔄 Future Enhancements

- [ ] Multi-language captions
- [ ] Video captioning frame-by-frame
- [ ] Custom model fine-tuning interface
- [ ] Batch upload in web interface
- [ ] Caption editing and feedback
- [ ] Database integration
- [ ] RESTful API
- [ ] Docker containerization
- [ ] Cloud deployment guide

## 📝 Example Output

### Input Image
![Sample Image](example.jpg)

### Generated Caption
```
"the image of a brown dog sitting on grass in a park during sunset"
```

### Batch Processing Output (captions.txt)
```
https://example.com/img1.jpg: the image of a laptop computer on a wooden desk
https://example.com/img2.jpg: the image of a person wearing a red jacket in the mountains
https://example.com/img3.jpg: the image of a coffee cup next to a notebook
```

## 📝 License

Educational and research purposes. Please respect image copyrights and website terms of service.

## 🙏 Acknowledgments

- Salesforce Research for BLIP
- Hugging Face for model hosting
- Gradio for the web framework
- Community contributors

---

**Happy Captioning! 🎨**
