import gradio as gr
from client import TryChroma

backend = TryChroma()


def image_search(image, index, category):
    constraints = {"subCategory": category} if len(category) > 0 else {}
    res = backend.search(image, constraints=constraints, index_name=index)
    return backend.parse_results(res, index)


demo = gr.Interface(fn=image_search,
                    inputs=[
                            gr.inputs.Image(shape=(60, 80), type="pil"),
                            gr.Radio(["image_clip", "image_fclip"]),
                            "text"
                    ],
                    outputs="gallery")
demo.launch()
