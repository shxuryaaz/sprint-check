"""Production entrypoint for gunicorn:  gunicorn wsgi:app

app.py's __main__ block (which calls init_client) does not run under gunicorn,
so initialize the OpenAI client here before the first request. Fails fast on
boot if OPENAI_API_KEY is unset."""
import loop1_pipeline as pipeline
import store
from app import app

pipeline.init_client()
store.init_db()  # create tables if a DATABASE_URL is configured (no-op otherwise)

if __name__ == "__main__":
    app.run()
