import modelbit as mb
import numpy as np
import gradio as gr
from dotenv import load_dotenv
from PIL import Image
import requests
import os
from pathlib import Path

load_dotenv()

mb.login()

image_dir = './images'
os.makedirs(image_dir,exist_ok=True)

gr.set_static_paths(paths=[image_dir])


with gr.Blocks().queue(default_concurrency_limit=10) as demo:
    with gr.Row() as row:
        with gr.Column() as col1:
            with gr.Tab("Upload Image"):
                input_image = gr.Image(label="Upload Image",type='filepath')
                image_submit_button = gr.Button("Confirm")
            with gr.Tab("Upload URL"):
                input_image_url = gr.Text(label="Upload Image URL")
                preview_image_box = gr.Image(visible=False,type='pil')
                with gr.Row():
                    image_url_preview_button = gr.Button("Preview")
                    image_url_submit_button = gr.Button("Confirm")

            with gr.Tab("Setting"):
                image_prompt = gr.Text(label="Prompt",value="What kind of music better suits this picture?")
                with gr.Row():
                    llava_num_token = gr.Slider(minimum=10,maximum=50,step=1,label="Prompt Length",value=30)
                    musicgen_num_token = gr.Slider(minimum=100,maximum=1000,step=10,label="Music Length",value=500)
                music_genre = gr.Text(label="Music Genre")


        with gr.Column() as col2:
            output_text = gr.Textbox()
            audio = gr.Audio(label="result",type="numpy")
            generate_new_music_button = gr.Button("Generate New Song",visible=False)




    @image_submit_button.click(inputs=[input_image])
    def save_image_to_folder(input_image):
        image_name = Path(input_image).name
        image = Image.open(input_image)
        image.save(os.path.join(image_dir,image_name))


    @image_url_preview_button.click(inputs=[input_image_url],outputs=[preview_image_box])
    def preview_image_url(url):
        image = Image.open(requests.get(url,stream=True).raw)
        return gr.Image(visible=True,value=image,label="Preview",container=True)
    
    def submit(input_image_url,image_prompt,llava_num_token,musicgen_num_token,music_genre):
        llava_result = llava_inference(input_image_url,image_prompt,llava_num_token)
        musicgen_result = musicgen_inference(llava_result + music_genre,musicgen_num_token)
        generate_new_music_button = gr.Button("Generate New Song",visible=True)
        return llava_result, (int(musicgen_result['sample_rate']), np.array(musicgen_result['audio'][0]).astype(np.float32)) , generate_new_music_button

    image_url_submit_button.click(submit,inputs=[input_image_url,image_prompt,llava_num_token,musicgen_num_token,music_genre],outputs=[output_text,audio,generate_new_music_button])
    
    generate_new_music_button.click(submit,inputs=[input_image_url,image_prompt,llava_num_token,musicgen_num_token,music_genre],outputs=[output_text,audio,generate_new_music_button])



def llava_inference(image_url,image_prompt,num_token):
    llava_response = mb.get_inference(
        workspace="haoenlu07",
        deployment="prompt_llava",
        data=[image_url, image_prompt, num_token]
    )
    return llava_response['data']

def musicgen_inference(prompt, num_token):
    musicgen_response = mb.get_inference(
        workspace="haoenlu07",
        deployment="musicgen",
        data=[prompt,num_token]
    )
    return musicgen_response['data']


demo.launch(server_name="0.0.0.0")
