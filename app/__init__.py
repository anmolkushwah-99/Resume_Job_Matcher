"""
Flask Application Factory.
Initializes the Flask app, CORS, configuration, error handlers, and route blueprints.
"""

import os
from flask import Flask, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def create_app(test_config=None) -> Flask:
    """
    Constructs and configures the core Flask application instance.
    """
    app = Flask(__name__, instance_relative_config=True)

    # Core Configurations
    app.config.from_mapping(
        SECRET_KEY=os.getenv("SECRET_KEY", "dev-secret-key-resume-matcher-2026"),
        MAX_CONTENT_LENGTH=10 * 1024 * 1024,  # 10 MB maximum request size
    )

    if test_config:
        app.config.from_mapping(test_config)

    # Enable Cross-Origin Resource Sharing (CORS)
    CORS(app)

    # Register blueprints
    from app.routes import api_bp
    app.register_blueprint(api_bp)

    # Custom Error Handlers
    @app.errorhandler(413)
    def request_entity_too_large(error):
        return jsonify({
            "success": False,
            "error": "File size exceeds the maximum allowed limit of 10 MB."
        }), 413

    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({
            "success": False,
            "error": getattr(error, "description", "Bad Request")
        }), 400

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            "success": False,
            "error": "The requested API endpoint was not found."
        }), 404

    @app.errorhandler(405)
    def method_not_allowed(error):
        return jsonify({
            "success": False,
            "error": "HTTP method not allowed for this endpoint."
        }), 405

    @app.errorhandler(500)
    def internal_server_error(error):
        return jsonify({
            "success": False,
            "error": "An internal server error occurred."
        }), 500

    return app


# Default application instance for WSGI / Flask CLI runners
app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
