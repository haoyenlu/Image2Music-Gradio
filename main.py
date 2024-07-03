import modelbit as mb
import numpy as np
import gradio as gr
from dotenv import load_dotenv
from PIL import Image
import requests
import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn
import threading

# FastAPI APP
app = FastAPI()

# Credential
load_dotenv()
mb.login()

# Static file
image_dir = Path('./images')
image_dir.mkdir(parents=True,exist_ok=True)

# Mount FastAPI StaticFiles server
app.mount("/images",StaticFiles(directory=image_dir),name="images")



with gr.Blocks().queue(default_concurrency_limit=10) as demo:
    
    use_upload_image = gr.State(True)

    with gr.Row() as row:
        with gr.Column() as col1:

            with gr.Tab("Upload Image") as tab1:
                tab1.select(fn=lambda x: True,inputs=[use_upload_image],outputs=[use_upload_image])
                input_image = gr.Image(label="Upload Image",type='filepath')
                image_submit_button = gr.Button("Confirm")

            with gr.Tab("Upload URL") as tab2:
                tab2.select(fn=lambda x: False,inputs=[use_upload_image],outputs=[use_upload_image])
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





    @image_submit_button.click(inputs=[input_image,image_prompt,llava_num_token,musicgen_num_token,music_genre],outputs=[output_text,audio,generate_new_music_button])
    def handle_image_upload(input_image,image_prompt,llava_num_token,musicgen_num_token,music_genre):
        if input_image is None:
            gr.Error('Please upload image first!')

        # Load and Save Image to static folder
        image_name = Path(input_image).name
        image = Image.open(input_image)
        image.save(os.path.join(image_dir,image_name))
        # Construct image url
        url = f"http://{os.environ['EC2_URL']}:{os.environ['EC2_PORT']}/images/{image_name}"
        return inference(url,image_prompt,llava_num_token,musicgen_num_token,music_genre)


    @image_url_preview_button.click(inputs=[input_image_url],outputs=[preview_image_box])
    def preview_image_url(url):
        image = Image.open(requests.get(url,stream=True).raw)
        return gr.Image(visible=True,value=image,label="Preview",container=True)
    
    def inference(input_image_url,image_prompt,llava_num_token,musicgen_num_token,music_genre):
        llava_result = llava_inference(input_image_url,image_prompt,llava_num_token)
        musicgen_result = musicgen_inference(llava_result + music_genre,musicgen_num_token)
        generate_new_music_button = gr.Button("Generate New Song",visible=True)
        return llava_result, (int(musicgen_result['sample_rate']), np.array(musicgen_result['audio'][0]).astype(np.float32)) , generate_new_music_button

    @image_url_submit_button.click(inputs=[input_image_url,image_prompt,llava_num_token,musicgen_num_token,music_genre],outputs=[output_text,audio,generate_new_music_button])
    def handle_image_url(input_image_url,image_prompt,llava_num_token,musicgen_num_token,music_genre):
        if input_image_url == "":
            gr.Error("Please Enter the URL of the image!")

        return inference(input_image_url,image_prompt,llava_num_token,musicgen_num_token,music_genre)

    @generate_new_music_button.click(inputs=[use_upload_image,input_image,input_image_url,image_prompt,llava_num_token,musicgen_num_token,music_genre],outputs=[output_text,audio,generate_new_music_button])
    def handle_generate_new_song(use_upload_image,input_image,input_image_url,image_prompt,llava_num_token,musicgen_num_token,music_genre):
        if use_upload_image:
            return handle_image_upload(input_image,image_prompt,llava_num_token,musicgen_num_token,music_genre)
        else:
            return handle_image_url(input_image_url,image_prompt,llava_num_token,musicgen_num_token,music_genre)


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


app = gr.mount_gradio_app(app,demo,path="/")

if __name__ == "__main__":
    config = uvicorn.Config(app=app,host="0.0.0.0",port=int(os.environ['EC2_PORT']))
    server = uvicorn.Server(config=config)
    thread = threading.Thread(target=server.run)
    thread.start()
