import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ACRCloud Configuration
ACR_ACCESS_KEY = os.getenv('ACR_ACCESS_KEY')
ACR_ACCESS_SECRET = os.getenv('ACR_ACCESS_SECRET')
ACR_HOST = os.getenv('ACR_HOST', 'identify-us.acrcloud.com')

# File settings
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
RESULTS_FOLDER = os.getenv('RESULTS_FOLDER', 'results')
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'flac', 'm4a', 'aac', 'ogg', 'wma'}
MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE', 50 * 1024 * 1024))

# Validation
if not ACR_ACCESS_KEY or not ACR_ACCESS_SECRET:
    raise ValueError("❌ Missing ACR_ACCESS_KEY or ACR_ACCESS_SECRET in .env file")