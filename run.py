from app.main import app  # Import app object from main.py
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render will provide PORT env var
    app.run(host='0.0.0.0', port=port)
