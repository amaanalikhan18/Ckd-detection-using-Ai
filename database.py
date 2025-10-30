from extensions import db
from models import User, Prediction, ModelPerformance, SystemLog

def init_db(app):
    """Initialize the database"""
    with app.app_context():
        # Create all tables
        db.create_all()
        
        # Create default admin user if it doesn't exist
        admin_user = User.query.filter_by(username='admin').first()
        if not admin_user:
            from werkzeug.security import generate_password_hash
            admin = User(
                username='admin',
                email='admin@kidneyclassifier.com',
                password_hash=generate_password_hash('admin123')  # Change this in production!
            )
            db.session.add(admin)
            db.session.commit()
            print("✅ Default admin user created (admin/admin123)")
        
        # Add initial model performance record
        model_perf = ModelPerformance.query.filter_by(model_version='resnet18_v1').first()
        if not model_perf:
            initial_perf = ModelPerformance(
                model_version='resnet18_v1',
                accuracy=0.95,  # Update with your actual model performance
                precision=0.94,
                recall=0.93,
                f1_score=0.94,
                total_predictions=0
            )
            db.session.add(initial_perf)
            db.session.commit()
            print("✅ Initial model performance record created")
        
        print("✅ Database initialized successfully")

def reset_db(app):
    """Reset the database (drop all tables and recreate)"""
    with app.app_context():
        db.drop_all()
        db.create_all()
        print("✅ Database reset successfully")

