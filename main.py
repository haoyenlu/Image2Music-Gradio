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
import yaml
import json

from inference import musicgen , llava

# FastAPI APP
app = FastAPI()

# Credential
load_dotenv()

# Static file
image_dir = Path('./images')
image_dir.mkdir(parents=True,exist_ok=True)

setting_file = './setting.yaml'

# Mount FastAPI StaticFiles server
app.mount("/images",StaticFiles(directory=image_dir),name="images")

with open(setting_file,'r') as file:
    setting = yaml.safe_load(file)

with gr.Blocks(theme=gr.themes.Base()).queue(default_concurrency_limit=10) as demo:
    
    image_url = gr.State("")

    gr.Markdown(
        """
        # Tiktok-techJam Challenge: Image-to-Music
        #### Upload an image from your device or enter an URL of an image and you will get your custom music based on the image!!!
        ---
        """
    )

    with gr.Row() as row:
        with gr.Column() as col1:

            with gr.Tab("Upload Image") as tab1:
                input_image = gr.Image(label="Upload Image",type='filepath')
                image_submit_button = gr.Button("Confirm")

            with gr.Tab("Upload URL") as tab2:
                input_image_url = gr.Text(label="Upload Image URL")
                preview_image_box = gr.Image(visible=False,type='pil')
                with gr.Row():
                    image_url_preview_button = gr.Button("Preview")
                    image_url_submit_button = gr.Button("Confirm")

            with gr.Tab("Setting"):
                image_prompt = gr.Text(label="Prompt",value="Describe the music that better suits this picture in a sentence.")
                with gr.Row():
                    llava_num_token = gr.Slider(minimum=10,maximum=50,step=1,label="Prompt Length",value=30)
                    musicgen_num_token = gr.Slider(minimum=100,maximum=1000,step=10,label="Music Length",value=500)
                with gr.Row():
                    genre_dropdown = gr.Dropdown(choices=setting['Genre'],max_choices=1,label="Genre")
                    mood_dropdown = gr.Dropdown(choices=setting['Mood'],max_choices=1,label="Mood")

                music_genre = gr.Text(label="Custom Music Specification")


        with gr.Column() as col2:
            output_text = gr.Textbox()
            audio = gr.Audio(label="result",type="numpy")
            generate_new_music_button = gr.Button("Generate New Song",visible=False)




    @image_submit_button.click(inputs=[input_image,image_prompt,llava_num_token,musicgen_num_token,music_genre],outputs=[image_url,output_text,audio,generate_new_music_button])
    def handle_image_upload(input_image,image_prompt,llava_num_token,musicgen_num_token,music_genre):
        if input_image is None:
            raise gr.Error('Please upload image first!')

        # Load and Save Image to static folder
        image_name = Path(input_image).name
        image = Image.open(input_image)
        image.save(os.path.join(image_dir,image_name))

        # Construct image url
        image_url = f"{os.environ['EC2_URL']}:{os.environ['GRADIO_PORT']}/images/{image_name}"

        return inference(image_url,image_prompt,llava_num_token,musicgen_num_token,music_genre)


    @image_url_preview_button.click(inputs=[input_image_url],outputs=[preview_image_box])
    def preview_image_url(url):
        image = Image.open(requests.get(url,stream=True).raw)
        return gr.Image(visible=True,value=image,label="Preview",container=True)
    
    def inference(image_url,image_prompt,llava_num_token,musicgen_num_token,music_genre):
        llava_result = llava_inference(image_url,image_prompt,llava_num_token)
        musicgen_result = musicgen_inference(llava_result + music_genre,musicgen_num_token)
        generate_new_music_button = gr.Button("Generate New Song",visible=True)
        return image_url,llava_result, (int(musicgen_result['sample_rate']), np.array(musicgen_result['audio'][0]).astype(np.float32)) , generate_new_music_button

    @image_url_submit_button.click(inputs=[input_image_url,image_prompt,llava_num_token,musicgen_num_token,music_genre],outputs=[image_url,output_text,audio,generate_new_music_button])
    def handle_image_url(input_image_url,image_prompt,llava_num_token,musicgen_num_token,music_genre):
        if input_image_url == "":
            raise gr.Error("Please Enter the URL of the image!")
        
        return inference(input_image_url,image_prompt,llava_num_token,musicgen_num_token,music_genre)

    @generate_new_music_button.click(inputs=[image_url,image_prompt,llava_num_token,musicgen_num_token,music_genre],outputs=[image_url,output_text,audio,generate_new_music_button])
    def handle_generate_new_song(image_url,image_prompt,llava_num_token,musicgen_num_token,music_genre):
        return inference(image_url,image_prompt,llava_num_token,musicgen_num_token,music_genre)
    


def llava_inference(image_url,image_prompt,num_token):
    # url = f"{os.environ['EC2_URL']}:{os.environ['MODEL_PORT']}/llava"
    data = {
        "url":image_url,
        "prompt": image_prompt,
        "max_num_token": num_token
    }
    return llava(**data)
    

def musicgen_inference(prompt, num_token):
    # url = f"{os.environ['EC2_URL']}:{os.environ['MODEL_PORT']}/musicgen"
    data = {
        "prompt": prompt,
        "max_num_token": num_token
    }
    return musicgen(**data)


app = gr.mount_gradio_app(app,demo,path="/")


if __name__ == "__main__":
    # config = uvicorn.Config(app=app,host="0.0.0.0",port=int(os.environ['GRADIO_PORT']))
    config = uvicorn.Config(app=app)
    server = uvicorn.Server(config=config)
    server.run()
    # thread = threading.Thread(target=server.run)
    # thread.start()
