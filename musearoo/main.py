from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to MuseAroo!"}

@app.get("/stdrummaroo_main.pyatus")
def get_status():
    return {"status": "ok"}
