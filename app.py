"""
Main Web Application Server Entry Point
AI-Based Resume Screening and Job Matching System
"""

import os
from app import create_app

app = create_app()

if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "True").lower() in ("true", "1", "yes")

    print("=" * 65)
    print(" AI-Based Resume Screening & Job Matching System")
    print(f" Web UI running at: http://{host}:{port}/")
    print(f" Dashboard URL:     http://{host}:{port}/dashboard")
    print(f" Health Check:      http://{host}:{port}/api/health")
    print("=" * 65)

    app.run(host=host, port=port, debug=debug)
