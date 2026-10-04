from pathlib import Path

from flask import Flask

from .config import AppConfig
from .database import init_db
from .routes import register_routes


def create_app(config_class=AppConfig):
    project_root = Path(__file__).resolve().parent.parent
    app = Flask(
        __name__,
        template_folder=str(project_root / "templates"),
        static_folder=str(project_root / "static"),
    )
    app.config.from_object(config_class)
    app.json.ensure_ascii = False

    init_db(app.config["DATABASE_PATH"])
    register_routes(app)
    return app
