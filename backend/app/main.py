from fastapi import FastAPI, Request
from fastapi.routing import APIRoute
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Favicon Static
# Get rid of favicon.ico 404 Not Found error
app.mount("/static", StaticFiles(directory="static"), name="static")

# routers example
# app.include_router(routers_name)

@app.get('/')
async def home(request: Request):
    domain = request.headers.get("host")
    return {"message": f"Hello from {domain}"}