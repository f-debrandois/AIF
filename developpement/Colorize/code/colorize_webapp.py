import gradio as gr
from PIL import Image
import requests
import io


def recognize_digit(image):
    image = Image.fromarray(image.astype('uint8'))
    img_binary = io.BytesIO()
    image.save(img_binary, format="PNG")
    # Send request to the API
    response = requests.post("http://colorize-api:5000/predict", data=img_binary.getvalue())
    return Image.open(io.BytesIO(response.content))

if __name__=='__main__':

    gr.Interface(fn=recognize_digit, 
                inputs="image", 
                outputs='image',
                live=True,
                description="Draw a number on the sketchpad to see the model's prediction.",
                ).launch(server_name="0.0.0.0", debug=True, share=True);