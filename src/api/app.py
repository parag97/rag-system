from fastapi import FastAPI

from src.api.routes import router
from src.api.ingestion_routes import router as ingestion_router
from src.api.dependencies import get_container


def create_app() -> FastAPI:
    app = FastAPI(
        title="RAG System",
        version="1.0.0",
    )

    @app.on_event("startup")
    def startup():

        container = get_container()

        #
        # force initialization
        #
        _ = container.rag_service

        #
        # start ingestion watcher
        #
        container.watcher_service.start()

    @app.on_event("shutdown")
    def shutdown():

        container = get_container()

        container.watcher_service.stop()

    app.include_router(router)
    app.include_router(ingestion_router)

    return app


app = create_app()