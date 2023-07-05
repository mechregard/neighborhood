import gradio as gr
from client import TryChroma

backend = TryChroma()


def image_search(image, index, category):
    constraints = {"subCategory": category} if category is not None and len(category) > 0 else {}
    res = backend.search(image, constraints=constraints, index_name=index)
    return backend.parse_results(res, index)


index_list = backend.indexes()

demo = gr.Interface(fn=image_search,
                    inputs=[
                            gr.inputs.Image(shape=(200, 270), type="pil"),
                            gr.Dropdown(index_list, label="Index to use"),
                            gr.Textbox(lines=1, label="SubCategory")
                    ],
                    outputs="gallery")
demo.launch(server_name="0.0.0.0", server_port=9090)
