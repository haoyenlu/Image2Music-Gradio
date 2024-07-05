import numpy as np
import gradio as gr
from dotenv import load_dotenv
from PIL import Image
import requests
import os
from pathlib import Path
from fastapi.staticfiles import StaticFiles
import threading
import yaml
import json

from inference import Pipeline

# model pipeline
pipeline = Pipeline()

# Credential
load_dotenv()

setting_file = './setting.yaml'


with open(setting_file,'r') as file:
    setting = yaml.safe_load(file)

with gr.Blocks(theme=gr.themes.Base()).queue(default_concurrency_limit=10) as demo:
    
    image = gr.State()

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
                input_image = gr.Image(label="Upload Image",type='numpy')
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
                    genre_dropdown = gr.Dropdown(choices=setting['Genre'],max_choices=1,label="Genre",value="None")
                    mood_dropdown = gr.Dropdown(choices=setting['Mood'],max_choices=1,label="Mood",value="None")

                music_description = gr.Text(label="Custom Music Specification")


        with gr.Column() as col2:
            output_text = gr.Textbox()
            audio = gr.Audio(label="result",type="numpy")
            generate_new_music_button = gr.Button("Generate New Song",visible=False)




    @image_submit_button.click(inputs=[input_image,image_prompt,llava_num_token,musicgen_num_token,music_description,genre_dropdown,mood_dropdown],
                               outputs=[image,output_text,audio,generate_new_music_button])
    def handle_image_upload(input_image,image_prompt,llava_num_token,musicgen_num_token,music_description,genre_dropdown,mood_dropdown):
        if input_image is None:
            raise gr.Error('Please upload image first!')

        return inference(input_image,image_prompt,llava_num_token,musicgen_num_token,music_description,genre_dropdown,mood_dropdown)


    @image_url_preview_button.click(inputs=[input_image_url],outputs=[preview_image_box])
    def preview_image_url(url):
        image = Image.open(requests.get(url,stream=True).raw)
        return gr.Image(visible=True,value=image,label="Preview",container=True)
    
    def inference(image,image_prompt,llava_num_token,musicgen_num_token,music_description, genre_dropdown, mood_dropdown):
        llava_result = llava_inference(image,image_prompt,llava_num_token)

        genre = "" if genre_dropdown == "None" else genre_dropdown
        mood = mood_dropdown if mood_dropdown != "None" else mood_dropdown
        musicgen_prompt = f"{llava_result},{music_description},{genre},{mood}"

        musicgen_result = musicgen_inference(f"{llava_result},{music_description},{genre_dropdown} in {mood_dropdown} mood", musicgen_num_token)
        generate_new_music_button = gr.Button("Generate New Song",visible=True)
        return image,llava_result, (int(musicgen_result['sample_rate']), np.array(musicgen_result['audio'][0]).astype(np.float32)) , generate_new_music_button

    @image_url_submit_button.click(inputs=[input_image_url,image_prompt,llava_num_token,musicgen_num_token,music_description,genre_dropdown,mood_dropdown],
                                   outputs=[image,output_text,audio,generate_new_music_button])
    def handle_image_url(input_image_url,image_prompt,llava_num_token,musicgen_num_token,music_description,genre_dropdown,mood_dropdown):
        if input_image_url == "":
            raise gr.Error("Please Enter the URL of the image!")
        
        image = Image.open(requests.get(input_image_url,stream=True).raw)
        return inference(image,image_prompt,llava_num_token,musicgen_num_token,music_description,genre_dropdown,mood_dropdown)

    @generate_new_music_button.click(inputs=[image,image_prompt,llava_num_token,musicgen_num_token,music_description,genre_dropdown,mood_dropdown],
                                     outputs=[image,output_text,audio,generate_new_music_button])
    def handle_generate_new_song(image,image_prompt,llava_num_token,musicgen_num_token,music_description,genre_dropdown,mood_dropdown):
        return inference(image,image_prompt,llava_num_token,musicgen_num_token,music_description,genre_dropdown,mood_dropdown)
    


def llava_inference(image,image_prompt,num_token):
    data = {
        "image":image,
        "prompt": image_prompt,
        "max_num_token": num_token
    }
    return pipeline.llava(**data)
    

def musicgen_inference(prompt, num_token):
    data = {
        "prompt": prompt,
        "max_num_token": num_token
    }
    return pipeline.musicgen(**data)


if __name__ == "__main__":
    demo.launch(share=True,server_name='0.0.0.0',server_port=int(os.environ['GRADIO_PORT']))