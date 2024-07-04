from fastapi import FastAPI
from fastapi.responses import JSONResponse
from typing import Union
from pydantic import BaseModel
import uvicorn
from PIL import Image
import requests
from transformers import AutoProcessor, MusicgenForConditionalGeneration, LlavaForConditionalGeneration, BitsAndBytesConfig
import os
import threading
from dotenv import load_dotenv
import logging
import accelerate
import bitsandbytes
import torch

load_dotenv()

app = FastAPI()


llava_model = LlavaForConditionalGeneration.from_pretrained("./llava-hf",local_files_only=True,device_map="auto")
llava_processor = AutoProcessor.from_pretrained("./llava-hf",local_files_only=True,device_map="auto")
musicgen_model = MusicgenForConditionalGeneration.from_pretrained("./musicgen-small",local_files_only=True,device_map="auto")
musicgen_processor = AutoProcessor.from_pretrained("./musicgen-small",local_files_only=True,device_map="auto")

class MusicGenRequestItem(BaseModel):
    prompt: str
    max_num_token: int 

class LlavaRequestItem(BaseModel):
    url: str
    prompt: str
    max_num_token: int 



@app.get('/health')
def health_check():
    return {"message":"Connect successfully."}


@app.post('/musicgen')
def musicgen(item: MusicGenRequestItem):
    inputs = musicgen_processor(text=[item.prompt],padding=True,return_tensors="pt",)
    result = musicgen_model.generate(**inputs, do_sample=True, guidance_scale=3,max_new_tokens=item.max_num_token)
    return JSONResponse(content={"audio":result[0].numpy().tolist(), "sample_rate":model.config.audio_encoder.sampling_rate})


@app.post('/llava')
def llava(item: LlavaRequestItem):
    image = Image.open(requests.get(item.url,stream=True).raw)
    full_prompt = f"USER: <image>\\n{item.prompt} ASSISTANT:"
    inputs = llava_processor(text=full_prompt, images=image, return_tensors="pt")
    generate_ids = llava_model.generate(**inputs, max_new_tokens=item.max_num_token)
    response = llava_processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].split("ASSISTANT:")[1]
    return response




if __name__ == '__main__':
    config = uvicorn.Config(app=app,host="0.0.0.0",port=int(os.environ['MODEL_PORT']))
    server = uvicorn.Server(config=config)
    thread = threading.Thread(target=server.run)
    thread.start()
