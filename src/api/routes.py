from fastapi import APIRouter
from fastapi import Depends

from src.api.schemas import QueryRequest
from src.api.schemas import QueryResponse

from src.api.dependencies import get_container

from src.container.application_container import (
    ApplicationContainer,
)

router = APIRouter()


@router.post(
    "/query",
    response_model=QueryResponse,
)
async def query(
    request: QueryRequest,
    container: ApplicationContainer = Depends(
        get_container
    ),
):
    answer = container.rag_service.answer(
        request.query
    )

    return QueryResponse(
        answer=answer,
    )