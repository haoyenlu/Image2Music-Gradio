import fastapi
import uvicorn
import threading

app = fastapi.FastAPI()


@app.get('/health')
def health_check():
    return "Hello World!"



if __name__ == '__main__':
    config = uvicorn.Config(app=app,port=8081)
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run)
    thread.start()