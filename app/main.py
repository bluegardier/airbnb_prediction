from fastapi import FastAPI
from .routers import houses

app = FastAPI()
app.include_router(houses.router)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
