from pydantic import BaseModel

class Contact(BaseModel):
    name: str | None = None
    url: str | None = None
    email: str | None = None

# Override FastAPI's default Contact model
import fastapi.openapi.models
fastapi.openapi.models.Contact = Contact
