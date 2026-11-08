"""Task 4.4 - production entry point: waitress WSGI server, Flask debug mode OFF.
    python -m app.serve        -> http://127.0.0.1:8000
"""
import logging
import os

from waitress import serve

from app.config import Config
from app.web import create_app

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
app = create_app()

if __name__ == "__main__":
    host, port = os.getenv("HOST", "127.0.0.1"), int(os.getenv("PORT", "8000"))
    logging.getLogger("visionguard").info("serving on http://%s:%s (debug off)", host, port)
    body_limit = int(Config.MAX_UPLOAD_MB * 2 * 1024 * 1024)   # above Flask's limit -> Flask shows the friendly 413 page
    serve(app, host=host, port=port, threads=int(os.getenv("THREADS", "4")),
          max_request_body_size=body_limit, inbuf_overflow=body_limit)   # whole request body stays in RAM
