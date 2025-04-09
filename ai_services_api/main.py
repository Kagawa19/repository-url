from ai_services_api.core.openapi import Contact
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse
from ai_services_api.controllers.chatbot_router import api_router as chatbot_router
from ai_services_api.controllers.search_router import api_router as search_router
from ai_services_api.controllers.recommendation_router import api_router as recommendation_router
from ai_services_api.controllers.message_router import api_router as message_router
from ai_services_api.services.chatbot.utils.redis_connection import redis_pool
from ai_services_api.controllers.publications_router import api_router as publications_router
from ai_services_api.controllers.autocomplete_router import api_router as autocomplete_router
from ai_services_api.controllers.analytics_router import api_router as analytics_router
# Add this to your application's startup code (e.g., in your main.py or __init__.py)
import logging
from ai_services_api.services.message.core.db_pool import get_connection_pool

logger = logging.getLogger(__name__)

# Explicitly initialize the pool during startup
logger.info("Initializing database connection pool...")
pool = get_connection_pool()
if pool:
    logger.info("Successfully initialized connection pool")
else:
    logger.error("Failed to initialize connection pool - will use direct connections")


# Define allowed GET routes
ALLOWED_GET_ROUTES = [
    "/",
    "/chatbot",
    "/recommendation", 
    "/search",
    "/content",
    "/health",
    "/chatbot/",
    "/recommendation/",
    "/search/",
    "/publications/",
    "/autocomplete/"
]

# Create the FastAPI app instance
app = FastAPI(
    title="AI Services Platform",
    version="0.0.1",
    contact=Contact(
        name="Brian Kimutai",
        email="briankimutai@icloud.com",
        url="https://your-url.com"
    )
)

# Middleware for GET request URL validation
@app.middleware("http")
async def validate_get_request(request: Request, call_next):
    # Only apply to GET requests
    if request.method == "GET":
        # Get the full path
        path = request.url.path
        
        # Check if the path starts with any of the allowed routes
        if not any(path.startswith(allowed_route.rstrip('/')) for allowed_route in ALLOWED_GET_ROUTES):
            # If not in allowed routes, raise a 403 Forbidden error
            raise HTTPException(status_code=403, detail="Access to this URL is not permitted")
    
    # Continue with the request if it passes validation
    response = await call_next(request)
    return response

# Shutdown event
async def shutdown_event():
    """Cleanup Redis connections on shutdown."""
    await redis_pool.close()

# Add shutdown event handler
app.add_event_handler("shutdown", shutdown_event)

# Configure CORS - Allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
    expose_headers=["Retry-After"]   # Important for rate limiting
)

# Include the API routers
app.include_router(chatbot_router, prefix="/chatbot")
app.include_router(recommendation_router, prefix="/recommendation")
app.include_router(search_router, prefix="/search")
app.include_router(message_router, prefix="/message")
app.include_router(publications_router, prefix="/publications")
app.include_router(autocomplete_router, prefix="/autocomplete")
app.include_router(analytics_router, prefix="/analytics")


# Routes with HTML responses
@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("ai_services_api/templates/index.html") as f:
        return f.read()

@app.get("/chatbot", response_class=HTMLResponse)
async def read_chatbot():
    with open("ai_services_api/templates/chatbot.html") as f:
        return f.read()

@app.get("/recommendation", response_class=HTMLResponse)
async def read_recommendation():
    with open("ai_services_api/templates/recommendations.html") as f:
        return f.read()

@app.get("/search", response_class=HTMLResponse)
async def read_search():
    with open("ai_services_api/templates/search.html") as f:
        return f.read()

@app.get("/content")
async def read_content():
    """Redirect to Streamlit dashboard"""
    return RedirectResponse(url="http://localhost:8501")

# Health check endpoint
@app.get("/health")
def health_check():
    """Service health check endpoint."""
    return {
        "status": "healthy",
        "services": {
            "api": "up",
            "redis": "up",
            "llm": "up"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)