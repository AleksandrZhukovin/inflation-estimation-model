from fastapi import Request
from fastapi import APIRouter
from fastapi.templating import Jinja2Templates

router = APIRouter(prefix="", tags=["Index"])

templates = Jinja2Templates(directory="server/templates")


@router.get("/")
def get_index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")
