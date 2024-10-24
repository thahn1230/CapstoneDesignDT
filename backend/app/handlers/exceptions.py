from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse

# HTTPException handler
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "message": exc.detail,
            "path": request.url.path
        },
    )

# 일반 예외 처리 핸들러 (500 Internal Server Error 같은 경우)
async def custom_general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "An unexpected error occurred.",
            "path": request.url.path
        },
    )