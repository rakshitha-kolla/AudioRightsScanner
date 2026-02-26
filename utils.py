import os
import json
from datetime import datetime
from config import UPLOAD_FOLDER, RESULTS_FOLDER, ALLOWED_EXTENSIONS

class FileHandler:
    """Handle file operations"""
    
    @staticmethod
    def validate_audio_file(file_path):
        """Check if file is valid audio"""
        if not os.path.exists(file_path):
            return False, "File not found"
        
        ext = file_path.split('.')[-1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            return False, f"Invalid format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        
        return True, "Valid"
    
    @staticmethod
    def ensure_folders():
        """Create required folders"""
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(RESULTS_FOLDER, exist_ok=True)


class ResultHandler:
    """Handle results storage and retrieval"""
    
    @staticmethod
    def save_result(file_name, detection_result):
        """Save detection result to JSON"""
        
        # Extract just the filename without path
        just_filename = os.path.basename(file_name)
        
        result_data = {
            'timestamp': datetime.now().isoformat(),
            'file_name': file_name,
            'copyrighted': detection_result.get('copyrighted'),
            'segments': detection_result.get('segments'),
            'error': detection_result.get('error'),
        }
        # Ensure results folder exists
        os.makedirs(RESULTS_FOLDER, exist_ok=True)
        
        result_file = os.path.join(RESULTS_FOLDER, f"{just_filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        with open(result_file, 'w') as f:
            json.dump(result_data, f, indent=2)
        
        return result_file
    
    @staticmethod
    def get_formatted_output(detection_result):         
        """Format result for display"""

        if detection_result.get('error'):
            return f"❌ Error: {detection_result['error']}"

        if detection_result.get('copyrighted') is None:
            return "⚠️  Unable to determine copyright status"

        if detection_result.get('copyrighted'):

            segments = detection_result.get('segments', [])

            if not segments:
                return "⚠️  Copyrighted music detected but no segment info available"

                output = "\n✓ COPYRIGHTED MUSIC DETECTED\n"
                output += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"

                for i, seg in enumerate(segments, 1):
                    music = seg.get("music", {})

                    output += f"""
                        Segment {i}
                        Title:    {music.get('title', 'Unknown')}
                        Artist:   {music.get('artist', 'Unknown')}
                        Album:    {music.get('album', 'Unknown')}
                        Start:    {round(seg.get('start', 0), 2)}s
                        End:      {round(seg.get('end', 0), 2)}s
                        Duration: {round(seg.get('duration', 0), 2)}s
                        ACR ID:   {music.get('acrid', 'N/A')}
                        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   
                        """

                return output

        else:
            return "✗ NO COPYRIGHTED MUSIC DETECTED - Safe to use"