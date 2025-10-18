import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import gradio as gr

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image preprocessing
transform_val_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                          [0.229, 0.224, 0.225])
])

# Model definition
resnet = models.resnet50(pretrained=False)
num_features = resnet.fc.in_features
resnet.fc = nn.Sequential(
    nn.Linear(num_features, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 1)
)

# Load trained weights
resnet.load_state_dict(torch.load("best_resnet_model.pth", map_location=device))
resnet = resnet.to(device)
resnet.eval()

# Prediction function
def predict_covid(image):
    img = transform_val_test(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = resnet(img)
        prob = torch.sigmoid(output).item()

    if prob > 0.5:
        # Pneumonia detected — elegant minimal style
        result = f"""
        <div style='
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
            font-family: "Segoe UI", sans-serif;
            color: #ff5252;
            background-color: rgba(255,82,82,0.1);
            border: 1px solid rgba(255,82,82,0.4);
            border-radius: 12px;
            padding: 15px 25px;
            font-size: 1.3em;
            font-weight: 600;
            text-align: center;
            width: 480px;
            box-shadow: 0 0 10px rgba(255,82,82,0.15);
        '>
            <span style="font-size:1.8em;">🫁</span>
            <div>
                Pneumonia Detected<br>
                <span style="font-size:0.85em; color:#ff8a80;">
                </span>
            </div>
        </div>
        """
    else:
        # Normal result — calm and positive
        result = f"""
        <div style='
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
            font-family: "Segoe UI", sans-serif;
            color: #66bb6a;
            background-color: rgba(102,187,106,0.1);
            border: 1px solid rgba(102,187,106,0.4);
            border-radius: 12px;
            padding: 15px 25px;
            font-size: 1.3em;
            font-weight: 600;
            text-align: center;
            width: 480px;
            box-shadow: 0 0 10px rgba(102,187,106,0.15);
        '>
            <span style="font-size:1.8em;">💚</span>
            <div>
                Normal — No Pneumonia<br>
                <span style="font-size:0.85em; color:#a5d6a7;">
                </span>
            </div>
        </div>
        """

    return result

# Custom CSS
custom_css = """
html, body, .gradio-container, #root {
    background: linear-gradient(135deg, #0d0d0d 0%, #1a1a1a 100%) !important;
    color: white !important;
    margin: 0;
    padding: 0;
    min-height: 100vh;
    width: 100%;
    overflow-y: auto;
}

body {
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    align-items: center;
    text-align: center;
}

h1 {
    text-align: center !important;
    margin-top: 20px !important;
    font-size: 2em !important;
    color: #ffffff !important;
    font-weight: 700 !important;
}

footer {display: none !important;}

/* Center and style image */
#component-1 img {
    border-radius: 15px;
    border: 3px solid #444;
    box-shadow: 0 0 10px rgba(255,255,255,0.2);
}

/* Buttons style */
button {
    border-radius: 10px !important;
}

/* Hide flag button completely */
button[data-testid="flag-button"] {
    display: none !important;
}

/* Center inputs and outputs vertically */
.gr-block.gr-box, .gradio-container > div {
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
}

/* Keep nice spacing */
.gradio-container > * {
    margin: 10px 0 !important;
}

/* Make result (output) centered and wider */
#component-2 {
    width: 500px !important;
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    text-align: center !important;
}
"""

# Gradio Interface
demo = gr.Interface(
    fn=predict_covid,
    inputs=gr.Image(type="pil", label= "Upload X-ray"),
    outputs=gr.HTML(label="🔍 AI Diagnosis"),
    title="COVID-19 X-ray Analyzer",
    description="<div style='font-size:1.1em; color:#ddd;'>AI model to detect potential signs of Pneumonia vs Normal lungs.</div>",
    theme="default",
    css=custom_css
)

demo.launch(server_name="127.0.0.1", server_port=7860)