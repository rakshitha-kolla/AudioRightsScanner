import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# AudD Configuration
AUDD_API_TOKEN = os.getenv('AUDD_API_TOKEN')
AUDD_API_URL = 'https://api.audd.io/'

# File settings
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
RESULTS_FOLDER = os.getenv('RESULTS_FOLDER', 'results')
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'flac', 'm4a', 'aac', 'ogg', 'wma'}
MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE', 50 * 1024 * 1024))

# Validation
if not AUDD_API_TOKEN:
    raise ValueError("❌ Missing AUDD_API_TOKEN in .env file")