import numpy as np
import gradio as gr
from dotenv import load_dotenv
from PIL import Image
import requests
import os
import threading
import yaml
import fastapi
import uvicorn


from inference import Pipeline

# model pipeline
app = fastapi.FastAPI()
pipeline = Pipeline()
pipeline.load_model()

# Credential
load_dotenv()

setting_file = './setting.yaml'

with open(setting_file,'r') as file:
    setting = yaml.safe_load(file)


def debug_fn():
    print("Debug")
    gr.Info("Debug Call")

with gr.Blocks(theme=gr.themes.Base()).queue(default_concurrency_limit=10) as demo:
    
    image = gr.State()
    audios = gr.State([])

    gr.Markdown(
        """
        # Image-to-Music 
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

            with gr.Tab("Setting",elem_id="setting"):
                with gr.Row():
                    image_prompt = gr.Text(label="Prompt",value="Describe the music that better suits this picture in a sentence..",scale=3)
                    num_song = gr.Number(value=1,minimum=1,maximum=5,scale=1)

                with gr.Row():
                    llava_num_token = gr.Slider(minimum=10,maximum=50,step=1,label="Prompt Length",value=30)
                    musicgen_num_token = gr.Slider(minimum=100,maximum=1000,step=10,label="Music Length",value=500)


                with gr.Row():
                    genre_dropdown = gr.Dropdown(choices=setting['Genre'],max_choices=1,label="Genre",value="None")
                    mood_dropdown = gr.Dropdown(choices=setting['Mood'],max_choices=1,label="Mood",value="None")

                music_description = gr.Text(label="Custom Music Specification")


        with gr.Column() as col2:
            output_text = gr.Textbox(label="AI",interactive=False,placeholder="Upload image to get a custom music!")

            @gr.render(inputs=[audios],triggers=[audios.change],trigger_mode='once')
            def dynamic_audio_component_render(audios):
                for i in range(len(audios)):
                    gr.Audio(value=(audios[i]),interactive=False,type="numpy",label=f"sample {i+1}")
                

            generate_new_music_button = gr.Button("Generate New Song",visible=False)



    @image_submit_button.click(inputs=[input_image,image_prompt,llava_num_token,musicgen_num_token,music_description,genre_dropdown,mood_dropdown,num_song],
                               outputs=[image,output_text,audios,generate_new_music_button])
    def handle_image_upload(input_image,image_prompt,llava_num_token,musicgen_num_token,music_description,genre_dropdown,mood_dropdown,num_song):
        if input_image is None:
            raise gr.Error('Please upload image first!')
        
        gr.Info("Generating Songs!")

        return inference(input_image,image_prompt,llava_num_token,musicgen_num_token,music_description,genre_dropdown,mood_dropdown,num_song)


    @image_url_preview_button.click(inputs=[input_image_url],outputs=[preview_image_box])
    def preview_image_url(url):
        image = Image.open(requests.get(url,stream=True).raw)
        return gr.Image(visible=True,value=image,label="Preview",container=True)
    
    def inference(image,image_prompt,llava_num_token,musicgen_num_token,music_description, genre_dropdown, mood_dropdown,num_song):
        llava_result = llava_inference(image,image_prompt,llava_num_token)

        genre = "" if genre_dropdown == "None" else genre_dropdown
        mood = mood_dropdown if mood_dropdown != "None" else mood_dropdown
        musicgen_prompt = f"{llava_result},{music_description},{genre},{mood}"

        musicgen_result = musicgen_inference(f"{llava_result},{musicgen_prompt}", musicgen_num_token,num_song)
        generate_new_music_button = gr.Button("Generate New Song",visible=True)

        sample_rate =  int(musicgen_result['sample_rate'])
        audio_result = musicgen_result['audio']

        audio_list = []
        for i in range(num_song):
            audio_list.append((sample_rate,np.array(audio_result[i,0,:]).astype(np.float32)))

        return image,llava_result, audio_list , generate_new_music_button

    @image_url_submit_button.click(inputs=[input_image_url,image_prompt,llava_num_token,musicgen_num_token,music_description,genre_dropdown,mood_dropdown,num_song],
                                   outputs=[image,output_text,audios,generate_new_music_button])
    def handle_image_url(input_image_url,image_prompt,llava_num_token,musicgen_num_token,music_description,genre_dropdown,mood_dropdown,num_song):
        if input_image_url == "":
            raise gr.Error("Please Enter the URL of the image!")
        
        gr.Info("Generating Songs!")
        image = Image.open(requests.get(input_image_url,stream=True).raw)
        return inference(image,image_prompt,llava_num_token,musicgen_num_token,music_description,genre_dropdown,mood_dropdown,num_song)

    @generate_new_music_button.click(inputs=[image,image_prompt,llava_num_token,musicgen_num_token,music_description,genre_dropdown,mood_dropdown,num_song],
                                     outputs=[image,output_text,audios,generate_new_music_button])
    def handle_generate_new_song(image,image_prompt,llava_num_token,musicgen_num_token,music_description,genre_dropdown,mood_dropdown,num_song):
        return inference(image,image_prompt,llava_num_token,musicgen_num_token,music_description,genre_dropdown,mood_dropdown,num_song)
    


def llava_inference(image,image_prompt,num_token):
    data = {
        "image":image,
        "prompt": image_prompt,
        "max_num_token": num_token
    }
    return pipeline.llava(**data)
    

def musicgen_inference(prompt, num_token, num_song):
    data = {
        "prompt": prompt,
        "max_num_token": num_token,
        "num_song": num_song
    }
    return pipeline.musicgen(**data)



app = gr.mount_gradio_app(app,demo,'/')



@app.get('/health')
def health_check():
    return {"message":"Connect successfully!"}



if __name__ == "__main__":
    
    config = uvicorn.Config(app=app,host="0.0.0.0",port=int(os.environ['GRADIO_PORT']))
    server = uvicorn.Server(config)
    # server.run()

    thread = threading.Thread(target=server.run)
    thread.start()
