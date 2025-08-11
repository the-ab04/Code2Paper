# backend/app.py
from flask import Flask
from flask_cors import CORS
from routes.paper_routes import paper_bp

def create_app():
    app = Flask(__name__)
    CORS(app)  # Enable CORS for all origins (for testing)
    app.register_blueprint(paper_bp)
    return app

app = create_app()

@app.route("/")
def home():
    return "Code2Paper backend running"

if __name__ == "__main__":
    # Use debug=False in production
    app.run(debug=True, port=5000, host="127.0.0.1")
