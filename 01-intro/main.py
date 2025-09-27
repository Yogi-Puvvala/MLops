from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def sayHello():
    return {"message": "Hello Yogi"}

@app.get("/aboutYou")
def aboutYou():
    return {"message": "I'm an GenAI aspirant"}