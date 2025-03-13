import torch
import torchvision.transforms as transforms
from flask import Flask, send_file, request
from PIL import Image
import io
from model import UNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

app = Flask(__name__)

MODEL_PATH = 'models/unet.pth'  # You can change this path if necessary

model = UNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda img: img.convert('L') if img.mode == 'RGB' else img),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image from the request
    img = Image.open(io.BytesIO(request.data))
    img = transform(img).unsqueeze(0).to(device)
    
    # Make the prediction
    with torch.no_grad():
        outputs = model(img)

    # Convert the prediction to a PIL Image
    pred_img = transforms.ToPILImage()(outputs.squeeze().cpu())

    # Save the image to a buffer
    buffer = io.BytesIO()
    pred_img.save(buffer, format='PNG')
    buffer.seek(0)

    return send_file(buffer, mimetype='image/png')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
