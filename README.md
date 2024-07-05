# Tiktok-Techjam Challenge - Image2Music 
Generative AI is a super strong tool that can generate basically anything. In this project, I utilized existing Gen AI model (Llava and Musicgen) to develop a web application that allow user to get a custom music based on the image. You can hear the picture!! :musical_note: :musical_note:

The UI of this project is built upon [Gradio](https://github.com/gradio-app/gradio.git), which is an open-source, light-weighted frontend library for quick and fast machine learning model demo.
The website is hosted using Vest.Ai GPU server. Try out the demo with this url: [Image2Music](http://91.150.160.38:1632/)


## Pipeline
![Image2Music](https://github.com/haoyenlu/image-to-music-app/assets/74141558/3e2536c2-7ee7-4427-ab53-dfe2e8fef83f)
First it feeds the image to the Llava model with a prompt that tells the model to generate sentence that describe the image, then the sentence is fed to the Musicgen model to generate the music.


### Demo
https://github.com/haoyenlu/image-to-music-app/assets/74141558/76f8fecd-1249-453c-b3cf-c8884fb39a1b

After user upload image to the website, click the confirm button, and it will automatically generate song based on the image.

---

https://github.com/haoyenlu/image-to-music-app/assets/74141558/dafe97e9-3fad-415f-886f-3c57c7d19fec

It also supports uploading url of the image and previewing the image in the website. The music will generate based on the image of the url.

---

https://github.com/haoyenlu/image-to-music-app/assets/74141558/5a1abf4e-dfc0-4146-ad9f-c003b378cbef

User can also specify the number of songs, the genre of the song, the mood of the song, and some custom specifications of the song.
Increase the prompt length can get a more accuracte description by the AI for better generation, and user can also specify the length of the song, the length of the song is roughly (Music Length / 50).

---

### Requirements
```
huggingface_hub
transformers
gradio
torch
fastapi
uvicorn
bitsandbytes
accelerate
```

### Install model weight
```
python download_model.py
```

### Run demo
```
python main.py
```
