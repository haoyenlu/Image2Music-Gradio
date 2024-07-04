from fastapi import FastAPI
from fastapi.responses import JSONResponse
from typing import Union
from pydantic import BaseModel
import uvicorn
from PIL import Image
import requests
from transformers import AutoProcessor, MusicgenForConditionalGeneration, LlavaForConditionalGeneration
from functools import cache
import os
import threading
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()


class MusicGenRequestItem(BaseModel):
    prompt: str
    max_num_token: int 

class LlavaRequestItem(BaseModel):
    url: str
    prompt: str
    max_num_token: int 

@cache
def load_llava_model():
    model = LlavaForConditionalGeneration.from_pretrained("./llava-hf",local_files_only=True)
    processor = AutoProcessor.from_pretrained("./llava-hf",local_files_only=True)
    return model , processor

@cache
def load_musicgen_model():
    model = MusicgenForConditionalGeneration.from_pretrained("./musicgen-small",local_files_only=True)
    processor = AutoProcessor.from_pretrained("./musicgen-small",local_files_only=True)
    return model , processor


@app.get('/health')
def health_check():
    return {"message":"Connect successfully."}


@app.post('/musicgen')
def musicgen(item: MusicGenRequestItem):
    model, processor = load_musicgen_model()
    inputs = processor(text=[item.prompt],padding=True,return_tensors="pt",)
    result = model.generate(**inputs, do_sample=True, guidance_scale=3,max_new_tokens=item.max_num_token)
    return JSONResponse(content={"audio":result[0], "sample_rate":model.config.audio_encoder.sampling_rate})


@app.post('/llava')
def llava(item: LlavaRequestItem):
    image = Image.open(requests.get(item.url,stream=True).raw)
    full_prompt = f"USER: <image>\\n{item.prompt} ASSISTANT:"
    model, processor = load_llava_model()
    inputs = processor(text=full_prompt, images=image, return_tensors="pt")
    generate_ids = model.generate(**inputs, max_new_tokens=item.max_num_token)
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].split("ASSISTANT:")[1]
    return response




if __name__ == '__main__':
    config = uvicorn.Config(app=app,host="0.0.0.0",port=int(os.environ['MODEL_PORT']))
    server = uvicorn.Server(config=config)
    thread = threading.Thread(target=server.run)
    thread.start()
