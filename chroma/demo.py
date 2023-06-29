import gradio as gr
from client import TryChroma
from PIL import Image


def search_images_from_image(image):
    img = Image.fromarray(image)
    cl = TryChroma()
    res = cl.search(img, constraints={}, index_name="image_clip")
    return cl.parse_results(res, "image_clip")


demo = gr.Interface(fn=search_images_from_image, inputs=gr.inputs.Image(shape=(60, 80)), outputs="gallery")
demo.launch()
