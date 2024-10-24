from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from mangum import Mangum

from app.routers.simulation_router import router as simulation_router
from app.handlers import exceptions

app = FastAPI()

# 커스텀 HTTPException 핸들러 등록
app.add_exception_handler(HTTPException, exceptions.custom_http_exception_handler)

# 일반 예외 처리 핸들러 등록 (선택 사항)
app.add_exception_handler(Exception, exceptions.custom_general_exception_handler)

# Favicon Static
# Get rid of favicon.ico 404 Not Found error
app.mount("/static", StaticFiles(directory="static"), name="static")

# routers
app.include_router(simulation_router)

# root router for testing
@app.get('/')
async def home(request: Request):
    domain = request.headers.get("host")
    return {"message": f"Hello from {domain}"}

handler = Mangum(app)