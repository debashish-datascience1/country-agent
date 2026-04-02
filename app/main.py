"""FastAPI application entry point."""
from __future__ import annotations

import logging
import logging.config
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import httpx
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.agent.graph import build_graph
from app.config import get_settings
from app.models import HealthResponse
from app.routers.ask import router as ask_router

STATIC_DIR = Path(__file__).parent / "static"


def _configure_logging(level: str) -> None:
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                }
            },
            "root": {
                "handlers": ["console"],
                "level": level,
            },
        }
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.
    - Startup: build the LangGraph once and create a shared httpx client.
    - Shutdown: close the HTTP client cleanly.
    """
    settings = get_settings()
    _configure_logging(settings.log_level)

    logger = logging.getLogger(__name__)
    logger.info("Starting Country Information Agent (model: %s)", settings.model_name)

    http_client = httpx.AsyncClient(timeout=settings.rest_countries_timeout)
    app.state.graph = build_graph(http_client)
    app.state.http_client = http_client

    logger.info("Agent ready")
    yield

    logger.info("Shutting down — closing HTTP client")
    await http_client.aclose()


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Country Information AI Agent",
        description=(
            "An AI agent powered by LangGraph and Groq that answers questions "
            "about countries using the REST Countries public API."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # Mount static assets (CSS, JS if any)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    # API routes
    app.include_router(ask_router, prefix="/api")

    @app.get("/", include_in_schema=False)
    async def root():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    async def health():
        return HealthResponse(status="ok", model=settings.model_name)

    return app


app = create_app()
