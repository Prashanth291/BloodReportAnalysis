# =====================================================
# BLOOD REPORT ANALYSIS API - Flask Backend (IMPROVED)
# =====================================================

from flask import Flask, request, jsonify, session, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import mysql.connector
from mysql.connector import Error
import google.generativeai as genai
import datetime
import os
import uuid
import json
from functools import wraps
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from PIL import Image
import re
from decimal import Decimal
from dotenv import load_dotenv
import logging
from typing import Optional, Dict, Any

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini API
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is required")

genai.configure(api_key=GEMINI_API_KEY)

app = Flask(__name__)

# Security Configuration
SECRET_KEY = os.environ.get('SECRET_KEY')
if not SECRET_KEY:
    raise ValueError("SECRET_KEY environment variable is required")

app.config.update(
    SECRET_KEY=SECRET_KEY,
    UPLOAD_FOLDER='uploads/',
    MAX_CONTENT_LENGTH=10 * 1024 * 1024,  # 10MB max file size
    PERMANENT_SESSION_LIFETIME=datetime.timedelta(days=7),
    SESSION_COOKIE_SECURE=True if os.environ.get('FLASK_ENV') == 'production' else False,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax'
)

# CORS Configuration
CORS(app, origins=['http://localhost:3000', 'http://127.0.0.1:5000'], 
     supports_credentials=True)

# Database configuration with validation
DB_PASSWORD = os.environ.get('DB_PASSWORD')
if not DB_PASSWORD:
    raise ValueError("DB_PASSWORD environment variable is required")

DB_CONFIG = {
    'host': os.environ.get('DB_HOST', 'localhost'),
    'database': os.environ.get('DB_NAME', 'blood_report_analysis'),
    'user': os.environ.get('DB_USER', 'root'),
    'password': DB_PASSWORD,
    'charset': 'utf8mb4',
    'collation': 'utf8mb4_unicode_ci',
    'autocommit': False,
    'raise_on_warnings': True
}

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# =====================================================
# DATABASE CONNECTION WITH CONNECTION POOLING
# =====================================================

def get_db_connection():
    """Create database connection with better error handling"""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            return connection
    except Error as e:
        logger.error(f"Database connection error: {e}")
        return None

def execute_query(query: str, params: tuple = None, fetch: str = None) -> Any:
    """Execute database query with proper error handling"""
    connection = get_db_connection()
    if not connection:
        return None
        
    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute(query, params or ())
        
        if fetch == 'one':
            result = cursor.fetchone()
        elif fetch == 'all':
            result = cursor.fetchall()
        else:
            connection.commit()
            result = cursor.rowcount
            
        return result
        
    except Error as e:
        logger.error(f"Query execution error: {e}")
        connection.rollback()
        return None
    finally:
        cursor.close()
        connection.close()

# =====================================================
# AUTHENTICATION UTILITIES (IMPROVED)
# =====================================================

def generate_session_token() -> str:
    """Generate a cryptographically secure session token"""
    import secrets
    return secrets.token_urlsafe(32)

def create_user_session(user_id: str) -> Optional[str]:
    """Create a new user session with improved security"""
    session_token = generate_session_token()
    expires_at = datetime.datetime.utcnow() + datetime.timedelta(days=7)
    
    query = """
        INSERT INTO user_sessions (user_id, session_token, expires_at, ip_address, user_agent, created_at)
        VALUES (%s, %s, %s, %s, %s, NOW())
    """
    
    result = execute_query(query, (
        user_id, session_token, expires_at,
        request.remote_addr, request.headers.get('User-Agent', '')[:500]
    ))
    
    return session_token if result else None

def verify_session_token(token: str) -> Optional[Dict]:
    """Verify session token and return user info"""
    query = """
        SELECT u.user_id, u.email, u.first_name, u.last_name, u.created_at
        FROM user_sessions us
        JOIN users u ON us.user_id = u.user_id
        WHERE us.session_token = %s 
        AND us.expires_at > NOW() 
        AND us.is_active = TRUE
    """
    
    user = execute_query(query, (token,), 'one')
    
    if user:
        # Update last accessed time
        update_query = "UPDATE user_sessions SET last_accessed = NOW() WHERE session_token = %s"
        execute_query(update_query, (token,))
        return user
    
    return None

def require_auth(f):
    """Enhanced authentication decorator"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization', '')
        
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({
                'success': False,
                'error': 'No valid authorization token provided'
            }), 401
        
        token = auth_header[7:]  # Remove 'Bearer ' prefix
        user = verify_session_token(token)
        
        if not user:
            return jsonify({
                'success': False,
                'error': 'Invalid or expired token'
            }), 401
        
        request.current_user = user
        return f(*args, **kwargs)
    
    return decorated_function

# =====================================================
# FRONTEND SERVING ROUTES (IMPROVED)
# =====================================================

@app.route('/')
def landing_page():
    """Serve the landing page"""
    try:
        return render_template('landing.html')
    except Exception as e:
        logger.error(f"Error serving landing page: {e}")
        return jsonify({'error': 'Page not found'}), 404

@app.route('/analyze')
def analyze_page():
    """Serve the analysis page"""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error serving analyze page: {e}")
        return jsonify({'error': 'Page not found'}), 404

# =====================================================
# AUTHENTICATION ROUTES (IMPROVED)
# =====================================================

@app.route('/api/auth/register', methods=['POST'])
def register():
    """Enhanced user registration endpoint"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        # Extract and validate input
        if 'name' in data:
            name_parts = data['name'].strip().split(' ', 1)
            first_name = name_parts[0]
            last_name = name_parts[1] if len(name_parts) > 1 else ''
        else:
            first_name = data.get('first_name', '').strip()
            last_name = data.get('last_name', '').strip()
        
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        
        # Enhanced validation
        if not all([email, password, first_name]):
            return jsonify({
                'success': False,
                'error': 'Email, password, and name are required'
            }), 400
        
        if not re.match(r'^[^\s@]+@[^\s@]+\.[^\s@]+$', email):
            return jsonify({
                'success': False,
                'error': 'Invalid email format'
            }), 400
        
        if len(password) < 8:
            return jsonify({
                'success': False,
                'error': 'Password must be at least 8 characters long'
            }), 400
        
        if len(first_name) < 2 or len(first_name) > 50:
            return jsonify({
                'success': False,
                'error': 'First name must be between 2 and 50 characters'
            }), 400
        
        # Check if email already exists
        existing_user = execute_query(
            "SELECT email FROM users WHERE email = %s", 
            (email,), 'one'
        )
        
        if existing_user:
            return jsonify({
                'success': False,
                'error': 'Email already registered'
            }), 409
        
        # Create new user
        user_id = str(uuid.uuid4())
        password_hash = generate_password_hash(password, method='pbkdf2:sha256')
        
        query = """
            INSERT INTO users (user_id, email, password_hash, first_name, last_name, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, NOW(), NOW())
        """
        
        result = execute_query(query, (user_id, email, password_hash, first_name, last_name))
        
        if not result:
            return jsonify({
                'success': False,
                'error': 'Failed to create account'
            }), 500
        
        # Create session
        session_token = create_user_session(user_id)
        if not session_token:
            return jsonify({
                'success': False,
                'error': 'Account created but login failed'
            }), 500
        
        logger.info(f"New user registered: {email}")
        
        return jsonify({
            'success': True,
            'message': 'Account created successfully',
            'user': {
                'user_id': user_id,
                'email': email,
                'first_name': first_name,
                'last_name': last_name,
                'full_name': f"{first_name} {last_name}".strip()
            },
            'token': session_token
        }), 201
        
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Server error occurred'
        }), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    """Enhanced user login endpoint"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        
        if not email or not password:
            return jsonify({
                'success': False,
                'error': 'Email and password are required'
            }), 400
        
        # Get user by email
        query = """
            SELECT user_id, email, password_hash, first_name, last_name, created_at
            FROM users WHERE email = %s AND is_active = TRUE
        """
        
        user = execute_query(query, (email,), 'one')
        
        if not user or not check_password_hash(user['password_hash'], password):
            return jsonify({
                'success': False,
                'error': 'Invalid email or password'
            }), 401
        
        # Create new session
        session_token = create_user_session(user['user_id'])
        if not session_token:
            return jsonify({
                'success': False,
                'error': 'Login failed'
            }), 500
        
        logger.info(f"User logged in: {email}")
        
        return jsonify({
            'success': True,
            'message': 'Login successful',
            'user': {
                'user_id': user['user_id'],
                'email': user['email'],
                'first_name': user['first_name'],
                'last_name': user['last_name'],
                'full_name': f"{user['first_name']} {user['last_name']}".strip()
            },
            'token': session_token
        }), 200
        
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Server error occurred'
        }), 500

@app.route('/api/auth/logout', methods=['POST'])
@require_auth
def logout():
    """Enhanced user logout endpoint"""
    try:
        token = request.headers.get('Authorization', '')[7:]  # Remove 'Bearer '
        
        query = "UPDATE user_sessions SET is_active = FALSE, updated_at = NOW() WHERE session_token = %s"
        execute_query(query, (token,))
        
        logger.info(f"User logged out: {request.current_user['email']}")
        
        return jsonify({
            'success': True,
            'message': 'Logout successful'
        }), 200
        
    except Exception as e:
        logger.error(f"Logout error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Server error occurred'
        }), 500

@app.route('/api/auth/profile', methods=['GET'])
@require_auth
def get_profile():
    """Get user profile information"""
    try:
        user = request.current_user
        return jsonify({
            'success': True,
            'user': {
                'user_id': user['user_id'],
                'email': user['email'],
                'first_name': user['first_name'],
                'last_name': user['last_name'],
                'full_name': f"{user['first_name']} {user['last_name']}".strip(),
                'member_since': user['created_at'].isoformat() if user['created_at'] else None
            }
        }), 200
    except Exception as e:
        logger.error(f"Profile fetch error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to fetch profile'
        }), 500

# =====================================================
# PARAMETER STATUS DETERMINATION (IMPROVED)
# =====================================================

def determine_parameter_status(parameter_name: str, value: float, user_gender: str = None) -> str:
    """Determine parameter status with database lookup"""
    query = """
        SELECT normal_range_min, normal_range_max, male_min, male_max, female_min, female_max
        FROM blood_parameters 
        WHERE LOWER(parameter_name) = LOWER(%s)
    """
    
    ranges = execute_query(query, (parameter_name,), 'one')
    
    if not ranges:
        return 'unknown'
    
    # Use gender-specific ranges if available
    if user_gender == 'male' and ranges.get('male_min') and ranges.get('male_max'):
        min_val, max_val = ranges['male_min'], ranges['male_max']
    elif user_gender == 'female' and ranges.get('female_min') and ranges.get('female_max'):
        min_val, max_val = ranges['female_min'], ranges['female_max']
    else:
        min_val, max_val = ranges['normal_range_min'], ranges['normal_range_max']
    
    if not min_val or not max_val:
        return 'unknown'
    
    # Determine status based on value
    if value < min_val * 0.7:
        return 'critical_low'
    elif value < min_val:
        return 'low'
    elif value > max_val * 1.5:
        return 'critical_high'
    elif value > max_val:
        return 'high'
    else:
        return 'normal'

# =====================================================
# FILE UPLOAD AND PROCESSING (IMPROVED)
# =====================================================

@app.route('/api/upload', methods=['POST'])
@require_auth
def upload_report():
    """Enhanced file upload and processing with Gemini API"""
    try:
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file uploaded'
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Enhanced file validation
        allowed_extensions = {'pdf', 'png', 'jpg', 'jpeg', 'tiff', 'bmp'}
        allowed_mime_types = {
            'application/pdf',
            'image/png', 'image/jpeg', 'image/jpg',
            'image/tiff', 'image/bmp'
        }
        
        file_extension = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        
        if file_extension not in allowed_extensions or file.mimetype not in allowed_mime_types:
            return jsonify({
                'success': False,
                'error': 'Invalid file type. Please upload PDF or image files only.'
            }), 400
        
        # Save file securely
        filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        file_size = os.path.getsize(file_path)
        user_id = request.current_user['user_id']
        
        # Create report record
        report_id = str(uuid.uuid4())
        
        query = """
            INSERT INTO blood_reports (
                report_id, user_id, original_filename, file_path, 
                file_size, file_type, report_status, created_at, updated_at
            ) VALUES (%s, %s, %s, %s, %s, %s, 'processing', NOW(), NOW())
        """
        
        result = execute_query(query, (
            report_id, user_id, file.filename,
            file_path, file_size, file_extension
        ))
        
        if not result:
            return jsonify({
                'success': False,
                'error': 'Failed to create report record'
            }), 500
        
        # Process with Gemini API
        try:
            file.seek(0)
            file_bytes = file.read()
            
            model = genai.GenerativeModel('gemini-2.5')
            
            uploaded_file = {
                'mime_type': file.mimetype,
                'data': file_bytes
            }
            
            prompt = """
            You are a medical report analysis expert. Extract blood test parameters from this medical report.
            
            Return ONLY a valid JSON object with the following structure. Do not include any markdown, explanations, or additional text:
            
            {
                "hemoglobin": <numeric_value_or_null>,
                "hematocrit": <numeric_value_or_null>,
                "wbc": <numeric_value_or_null>,
                "rbc": <numeric_value_or_null>,
                "platelet": <numeric_value_or_null>,
                "glucose": <numeric_value_or_null>,
                "cholesterol": <numeric_value_or_null>,
                "hdl": <numeric_value_or_null>,
                "ldl": <numeric_value_or_null>,
                "triglycerides": <numeric_value_or_null>,
                "creatinine": <numeric_value_or_null>,
                "urea": <numeric_value_or_null>,
                "sodium": <numeric_value_or_null>,
                "potassium": <numeric_value_or_null>,
                "alt": <numeric_value_or_null>,
                "ast": <numeric_value_or_null>,
                "bilirubin": <numeric_value_or_null>,
                "tsh": <numeric_value_or_null>,
                "t4": <numeric_value_or_null>,
                "vitamin_d": <numeric_value_or_null>,
                "b12": <numeric_value_or_null>,
                "iron": <numeric_value_or_null>
            }
            
            Extract only numeric values. If a parameter is not found or unclear, use null.
            """
            
            response = model.generate_content([prompt, uploaded_file])
            response_text = response.text.strip()
            
            # Clean response
            response_text = re.sub(r'^```json\s*', '', response_text)
            response_text = re.sub(r'\s*```$', '', response_text)
            
            try:
                parameters = json.loads(response_text)
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON response from AI model")
            
            # Filter out null values and validate
            valid_parameters = {}
            for key, value in parameters.items():
                if value is not None and isinstance(value, (int, float)) and value >= 0:
                    valid_parameters[key] = float(value)
            
            if not valid_parameters:
                # Update report status
                execute_query(
                    "UPDATE blood_reports SET report_status = 'failed', notes = %s, updated_at = NOW() WHERE report_id = %s",
                    ('No valid blood parameters found', report_id)
                )
                return jsonify({
                    'success': False,
                    'error': 'No blood parameters could be extracted from the report'
                }), 400
            
            # Save parameters to database
            if save_parameters_to_db(report_id, valid_parameters, user_id):
                execute_query(
                    "UPDATE blood_reports SET report_status = 'completed', processed_at = NOW(), updated_at = NOW() WHERE report_id = %s",
                    (report_id,)
                )
                
                logger.info(f"Report processed successfully: {report_id}")
                
                return jsonify({
                    'success': True,
                    'message': 'Report processed successfully',
                    'report_id': report_id,
                    'parameters_found': len(valid_parameters),
                    'parameters': valid_parameters
                }), 200
            else:
                execute_query(
                    "UPDATE blood_reports SET report_status = 'failed', notes = %s, updated_at = NOW() WHERE report_id = %s",
                    ('Failed to save parameters', report_id)
                )
                return jsonify({
                    'success': False,
                    'error': 'Failed to save extracted parameters'
                }), 500
                
        except Exception as e:
            logger.error(f"Processing error for report {report_id}: {str(e)}")
            execute_query(
                "UPDATE blood_reports SET report_status = 'failed', notes = %s, updated_at = NOW() WHERE report_id = %s",
                (f'Processing failed: {str(e)}', report_id)
            )
            return jsonify({
                'success': False,
                'error': 'Failed to process report. Please ensure the image is clear and contains blood test results.'
            }), 500
            
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Upload failed'
        }), 500

def save_parameters_to_db(report_id: str, parameters: Dict[str, float], user_id: str) -> bool:
    """Save extracted parameters to database with improved error handling"""
    try:
        connection = get_db_connection()
        if not connection:
            return False
            
        cursor = connection.cursor(dictionary=True)
        
        # Get parameter definitions
        cursor.execute("SELECT parameter_id, parameter_name FROM blood_parameters")
        param_definitions = {
            row['parameter_name'].lower().replace(' ', '_'): row['parameter_id'] 
            for row in cursor.fetchall()
        }
        
        # Get user profile for gender-specific ranges
        cursor.execute("SELECT gender FROM users WHERE user_id = %s", (user_id,))
        user_data = cursor.fetchone()
        user_gender = user_data.get('gender') if user_data else None
        
        saved_count = 0
        for param_name, value in parameters.items():
            if param_name in param_definitions:
                parameter_id = param_definitions[param_name]
                status = determine_parameter_status(param_name, value, user_gender)
                
                query = """
                    INSERT INTO blood_parameter_values 
                    (report_id, parameter_id, value, status, confidence_score, created_at)
                    VALUES (%s, %s, %s, %s, %s, NOW())
                """
                cursor.execute(query, (report_id, parameter_id, value, status, 0.90))
                saved_count += 1
        
        connection.commit()
        logger.info(f"Saved {saved_count} parameters for report {report_id}")
        return saved_count > 0
        
    except Exception as e:
        logger.error(f"Error saving parameters: {str(e)}")
        if connection:
            connection.rollback()
        return False
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'connection' in locals():
            connection.close()

# =====================================================
# ERROR HANDLERS (IMPROVED)
# =====================================================

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({
        'success': False,
        'error': 'Resource not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

@app.errorhandler(413)
def too_large(error):
    return jsonify({
        'success': False,
        'error': 'File too large. Maximum size is 10MB.'
    }), 413

@app.errorhandler(400)
def bad_request(error):
    return jsonify({
        'success': False,
        'error': 'Bad request'
    }), 400

# =====================================================
# HEALTH CHECK ENDPOINT
# =====================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        connection = get_db_connection()
        if connection:
            connection.close()
            db_status = 'connected'
        else:
            db_status = 'disconnected'
        
        return jsonify({
            'success': True,
            'status': 'healthy',
            'database': db_status,
            'timestamp': datetime.datetime.utcnow().isoformat()
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            'success': False,
            'status': 'unhealthy',
            'error': str(e)
        }), 500

# =====================================================
# MAIN APPLICATION
# =====================================================

if __name__ == '__main__':
    # Validate required environment variables on startup
    required_env_vars = ['SECRET_KEY', 'DB_PASSWORD', 'GEMINI_API_KEY']
    missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    logger.info("Starting BloodSight API server...")
    
    # Development vs Production settings
    if os.environ.get('FLASK_ENV') == 'production':
        app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
    else:
        app.run(debug=True, host='0.0.0.0', port=5000)