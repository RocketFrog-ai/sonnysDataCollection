import uvicorn
from fastapi import FastAPI
from app.site_analysis.server.routes import router as site_analysis_router
from app.pnl_analysis.server.central_routes import router as central_input_router
from app.pnl_analysis.server.routes import router as pnl_analysis_router
from app.utils import common as calib
from fastapi.middleware.cors import CORSMiddleware


FAST_API_HOST = calib.FAST_API_HOST
FAST_API_PORT = calib.FAST_API_PORT

# Create FastAPI application
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(site_analysis_router, prefix="/v1")
app.include_router(central_input_router, prefix="/v1")
app.include_router(pnl_analysis_router, prefix="/v1")

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with basic service information"""
    return {
        "status": "running"
    }

if __name__ == "__main__":
    uvicorn.run(app, host=FAST_API_HOST, port=int(FAST_API_PORT))
