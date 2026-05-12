"""
API Key authentication dependency.

Usage in a router:
    from api.middleware.auth import require_api_key

    @router.post("/endpoint")
    async def my_endpoint(
        _: str = Depends(require_api_key),
        ...
    ):
        ...
"""

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from api.config import Settings

settings = Settings()

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(api_key: str = Security(_api_key_header)) -> str:
    """
    Dependency that validates the X-API-Key header.

    Returns the validated key, or raises 401 Unauthorized.
    """
    if not api_key or api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Include 'X-API-Key: <key>' header.",
        )
    return api_key
