"""
FastAPI application factory.

Creates the app with CORS middleware, static file mounting
for job artifacts, and experiment routers.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from server.routers.index import router as index_router
from server.routers.experiments import router as experiments_router


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure outputs/ exists before mounting
outputs_dir = Path("outputs")
outputs_dir.mkdir(parents=True, exist_ok=True)

app.mount(
    "/static/jobs",
    StaticFiles(directory=str(outputs_dir)),
    name="job_artifacts",
)

app.include_router(index_router)
app.include_router(experiments_router)
