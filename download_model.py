from huggingface_hub import snapshot_download

snapshot_download(repo_id="llava-hf/llava-1.5-7b-hf", local_dir="./llava-hf")
snapshot_download(repo_id="facebook/musicgen-small", local_dir="./musicgen-small")