from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request

from app.errors import ApiError
from app.schemas import (
    DeleteResponse,
    DocumentCreate,
    DocumentList,
    DocumentOut,
    DocumentUpdate,
)
from app.security import require_client_id

router = APIRouter(prefix="/documents", tags=["documents"])



def not_found() -> ApiError:
    # 404 (not 403) for other people's documents: do not reveal that the id exists
    return ApiError(404, "not_found", "Document not found.")


@router.get("", response_model=DocumentList, summary="List my documents (newest first)")
def list_documents(
    request: Request,
    limit: int = Query(50, ge=1, le=100),
    skip: int = Query(0, ge=0),
    owner_id: str = Depends(require_client_id),
):
    docs, total = request.app.state.store.list_documents(owner_id, limit, skip)
    return DocumentList(documents=docs, count=len(docs), total=total)


@router.post("", response_model=DocumentOut, status_code=201, summary="Create a document")
def create_document(body: DocumentCreate, request: Request, owner_id: str = Depends(require_client_id)):
    store = request.app.state.store
    limit = request.app.state.settings.max_documents_per_client
    if store.count_documents(owner_id) >= limit:
        raise ApiError(409, "document_limit_reached", f"You can store at most {limit} documents. Delete one first.")
    return store.create_document(owner_id, body.title, body.content)


@router.get("/{doc_id}", response_model=DocumentOut, summary="Get one of my documents")
def get_document(doc_id: str, request: Request, owner_id: str = Depends(require_client_id)):
    doc = request.app.state.store.get_document(owner_id, doc_id)
    if doc is None:
        raise not_found()
    return doc


@router.put("/{doc_id}", response_model=DocumentOut, summary="Update title and/or content")
def update_document(doc_id: str, body: DocumentUpdate, request: Request, owner_id: str = Depends(require_client_id)):
    doc = request.app.state.store.update_document(owner_id, doc_id, body.title, body.content)
    if doc is None:
        raise not_found()
    return doc


@router.delete("/{doc_id}", response_model=DeleteResponse, summary="Delete a document")
def delete_document(doc_id: str, request: Request, owner_id: str = Depends(require_client_id)):
    if not request.app.state.store.delete_document(owner_id, doc_id):
        raise not_found()
    return DeleteResponse(deleted=True, id=doc_id)
