#init_db.py
from app import app
from backend.models import db

with app.app_context():
    db.create_all()
    print("✅ Base de données SQLite créée !")
#__________________________________________________________________________________

