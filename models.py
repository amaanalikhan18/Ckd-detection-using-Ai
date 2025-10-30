from extensions import db
from datetime import datetime
from sqlalchemy.sql import func

class User(db.Model):
    """User model for authentication"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    
    # Relationship with predictions
    predictions = db.relationship('Prediction', backref='user', lazy=True, cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<User {self.username}>'
    
    def get_prediction_count(self):
        """Get total number of predictions by this user"""
        return len(self.predictions)
    
    def get_recent_predictions(self, limit=5):
        """Get recent predictions by this user"""
        return Prediction.query.filter_by(user_id=self.id)\
                              .order_by(Prediction.created_at.desc())\
                              .limit(limit).all()
    
    def get_accuracy_stats(self):
        """Get user's prediction statistics (mock data for now)"""
        predictions = self.predictions
        if not predictions:
            return {'total': 0, 'by_class': {}}
        
        stats = {
            'total': len(predictions),
            'by_class': {},
            'confidence_avg': 0
        }
        
        class_counts = {}
        total_confidence = 0
        
        for pred in predictions:
            class_name = pred.predicted_class
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            total_confidence += pred.confidence
        
        stats['by_class'] = class_counts
        stats['confidence_avg'] = total_confidence / len(predictions) if predictions else 0
        
        return stats

class Prediction(db.Model):
    """Model to store prediction results"""
    __tablename__ = 'predictions'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255))
    predicted_class = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    class_probabilities = db.Column(db.Text)  # JSON string of all class probabilities
    processing_time = db.Column(db.Float)  # Time taken for prediction in seconds
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Additional metadata
    image_width = db.Column(db.Integer)
    image_height = db.Column(db.Integer)
    file_size = db.Column(db.Integer)  # in bytes
    notes = db.Column(db.Text)  # User notes about the prediction
    
    def __repr__(self):
        return f'<Prediction {self.id}: {self.predicted_class} ({self.confidence:.2f})>'
    
    def get_formatted_date(self):
        """Get formatted creation date"""
        return self.created_at.strftime('%Y-%m-%d %H:%M')
    
    def get_confidence_percentage(self):
        """Get confidence as percentage"""
        return f"{self.confidence * 100:.1f}%"
    
    def get_class_probabilities_dict(self):
        """Convert class probabilities JSON string to dict"""
        if self.class_probabilities:
            import json
            try:
                return json.loads(self.class_probabilities)
            except:
                return {}
        return {}
    
    def is_high_confidence(self, threshold=0.8):
        """Check if prediction has high confidence"""
        return self.confidence >= threshold
    
    def get_risk_level(self):
        """Get risk level based on predicted class"""
        risk_levels = {
            'Normal': 'Low',
            'Cyst': 'Low-Medium',
            'Stone': 'Medium',
            'Tumor': 'High'
        }
        return risk_levels.get(self.predicted_class, 'Unknown')
    
    def get_recommendation(self):
        """Get medical recommendation based on prediction"""
        recommendations = {
            'Normal': 'No immediate action required. Continue regular check-ups.',
            'Cyst': 'Monitor with follow-up imaging. Consult urologist if symptoms develop.',
            'Stone': 'Consult urologist. May require treatment depending on size and symptoms.',
            'Tumor': 'Urgent urological consultation required. Further imaging and biopsy may be needed.'
        }
        return recommendations.get(self.predicted_class, 'Consult with a healthcare professional.')

class ModelPerformance(db.Model):
    """Track model performance metrics over time"""
    __tablename__ = 'model_performance'
    
    id = db.Column(db.Integer, primary_key=True)
    model_version = db.Column(db.String(50), nullable=False)
    accuracy = db.Column(db.Float)
    precision = db.Column(db.Float)
    recall = db.Column(db.Float)
    f1_score = db.Column(db.Float)
    total_predictions = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<ModelPerformance {self.model_version}: Acc={self.accuracy:.3f}>'

class SystemLog(db.Model):
    """System logs for monitoring and debugging"""
    __tablename__ = 'system_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    level = db.Column(db.String(20), nullable=False)  # INFO, WARNING, ERROR
    message = db.Column(db.Text, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    prediction_id = db.Column(db.Integer, db.ForeignKey('predictions.id'), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<SystemLog {self.level}: {self.message[:50]}>'
    
    @staticmethod
    def log_info(message, user_id=None, prediction_id=None):
        """Log info message"""
        log = SystemLog(
            level='INFO',
            message=message,
            user_id=user_id,
            prediction_id=prediction_id
        )
        db.session.add(log)
        db.session.commit()
    
    @staticmethod
    def log_warning(message, user_id=None, prediction_id=None):
        """Log warning message"""
        log = SystemLog(
            level='WARNING',
            message=message,
            user_id=user_id,
            prediction_id=prediction_id
        )
        db.session.add(log)
        db.session.commit()
    
    @staticmethod
    def log_error(message, user_id=None, prediction_id=None):
        """Log error message"""
        log = SystemLog(
            level='ERROR',
            message=message,
            user_id=user_id,
            prediction_id=prediction_id
        )
        db.session.add(log)
        db.session.commit()

# Database utility functions
def get_prediction_statistics():
    """Get overall prediction statistics"""
    from sqlalchemy import func
    
    stats = {}
    
    # Total predictions
    stats['total_predictions'] = db.session.query(Prediction).count()
    
    # Predictions by class
    class_stats = db.session.query(
        Prediction.predicted_class,
        func.count(Prediction.id).label('count'),
        func.avg(Prediction.confidence).label('avg_confidence')
    ).group_by(Prediction.predicted_class).all()
    
    stats['by_class'] = {}
    for class_name, count, avg_conf in class_stats:
        stats['by_class'][class_name] = {
            'count': count,
            'avg_confidence': float(avg_conf) if avg_conf else 0
        }
    
    # Predictions per day (last 30 days)
    from datetime import datetime, timedelta
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    
    daily_stats = db.session.query(
        func.date(Prediction.created_at).label('date'),
        func.count(Prediction.id).label('count')
    ).filter(
        Prediction.created_at >= thirty_days_ago
    ).group_by(
        func.date(Prediction.created_at)
    ).all()
    
    stats['daily'] = [{'date': str(date), 'count': count} for date, count in daily_stats]
    
    # Average confidence
    avg_confidence = db.session.query(func.avg(Prediction.confidence)).scalar()
    stats['avg_confidence'] = float(avg_confidence) if avg_confidence else 0
    
    return stats

def get_user_statistics():
    """Get user registration statistics"""
    stats = {}
    
    # Total users
    stats['total_users'] = db.session.query(User).count()
    
    # Active users (users with at least one prediction)
    active_users = db.session.query(User.id).join(Prediction).distinct().count()
    stats['active_users'] = active_users
    
    # Users registered in last 30 days
    from datetime import datetime, timedelta
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    recent_users = db.session.query(User).filter(User.created_at >= thirty_days_ago).count()
    stats['recent_users'] = recent_users
    
    return stats