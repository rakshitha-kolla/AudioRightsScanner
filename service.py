import requests
import base64
import hmac
import hashlib
from datetime import datetime
import json
import subprocess
import os
import tempfile

class ACRCloudService:
    """Service to identify copyrighted music using ACRCloud API"""
    
    def __init__(self, access_key, access_secret, host='identify-us.acrcloud.com'):
        self.access_key = access_key
        self.access_secret = access_secret
        self.host = host
    
    def _trim_audio(self, audio_file_path, duration_seconds=10):
        """
        Trim audio file to specified duration using ffmpeg
        
        Args:
            audio_file_path (str): Path to audio file
            duration_seconds (int): Duration to trim to (default 15 seconds)
        
        Returns:
            bytes: Trimmed audio data
        """
        try:
            # Create temp file for trimmed audio
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                tmp_path = tmp.name
            
            # Use ffmpeg to trim audio
            cmd = [
                'ffmpeg',
                '-i', audio_file_path,
                '-t', str(duration_seconds),
                '-q:a', '9',
                '-acodec', 'libmp3lame',
                tmp_path,
                '-y'  # Overwrite without asking
            ]
            
            subprocess.run(cmd, capture_output=True, check=True)
            
            # Read trimmed audio
            with open(tmp_path, 'rb') as f:
                trimmed_data = f.read()
            
            # Clean up temp file
            os.unlink(tmp_path)
            
            return trimmed_data
        
        except FileNotFoundError:
            print("⚠️  ffmpeg not installed. Using original file (may fail if too large)")
            with open(audio_file_path, 'rb') as f:
                return f.read()
        except Exception as e:
            print(f"⚠️  Could not trim audio: {e}. Using original file")
            with open(audio_file_path, 'rb') as f:
                return f.read()
    
    def identify(self, audio_file_path,skip_trim=False):
        """
        Identify if audio contains copyrighted music
        
        Args:
            audio_file_path (str): Path to audio file
        
        Returns:
            dict: {
                'copyrighted': bool,
                'music': dict (if copyrighted),
                'error': str (if error)
            }
        """
        
        try:
            # Trim audio to 15 seconds
            if not skip_trim:
                print("✂️  Trimming audio to 15 seconds...")
                audio_data = self._trim_audio(audio_file_path, duration_seconds=10)
            else:
                print("⚠️  Skipping trim (using original file)")
                with open(audio_file_path, 'rb') as f:
                    audio_data = f.read()
            
            print(f"📦 Trimmed file size: {len(audio_data) / 1024 / 1024:.2f} MB")
            
            # Generate signature
            timestamp = str(int(datetime.now().timestamp()))
            signature_string = f"POST\n/v1/identify\n{self.access_key}\naudio\n1\n{timestamp}"
            
            signature = base64.b64encode(
                hmac.new(
                    self.access_secret.encode(),
                    signature_string.encode(),
                    hashlib.sha1
                ).digest()
            ).decode()
            
            # Prepare request
            files = {
                'sample': audio_data,
                'access_key': (None, self.access_key),
                'timestamp': (None, timestamp),
                'signature': (None, signature),
                'data_type': (None, 'audio'),
                'signature_version': (None, '1'),
            }
            
            # Send request
            url = f'https://{self.host}/v1/identify'
            print(f"🔗 Sending to ACRCloud...")
            
            response = requests.post(url, files=files, timeout=30)
            print(f"📡 Response Status: {response.status_code}")
            
            result = response.json()
            
            return self._parse_result(result)
        
        except FileNotFoundError:
            return {'copyrighted': None, 'error': 'File not found'}
        except requests.exceptions.RequestException as e:
            return {'copyrighted': None, 'error': f'Network error: {str(e)}'}
        except Exception as e:
            return {'copyrighted': None, 'error': f'Error: {str(e)}'}
    
    def _parse_result(self, result):
        """Parse ACRCloud API response"""
        
        # Handle status object response
        if 'status' in result:
            status = result['status']
            if status.get('code') == 0:
                if 'metadata' in result and 'music' in result['metadata']:
                    music_list = result['metadata']['music']
                    if music_list:
                        music = music_list[0]
                        return {
                            'copyrighted': True,
                            'music': {
                                'title': music.get('title'),
                                'artist': music['artists'][0]['name'] if music.get('artists') else 'Unknown',
                                'album': music.get('album', {}).get('name', 'Unknown'),
                                'duration': music.get('duration_ms', 0) // 1000,
                                'acrid': music.get('acrid'),
                            }
                        }
            elif status.get('code') == 1:
                return {'copyrighted': False}
            else:
                error_msg = status.get('msg', 'Unknown error')
                error_code = status.get('code', 'N/A')
                return {'copyrighted': None, 'error': f'API Error ({error_code}): {error_msg}'}
        
        # Handle direct code response
        if 'code' in result:
            if result.get('code') == 0:
                if 'metadata' in result and 'music' in result['metadata']:
                    music_list = result['metadata']['music']
                    if music_list:
                        music = music_list[0]
                        return {
                            'copyrighted': True,
                            'music': {
                                'title': music.get('title'),
                                'artist': music['artists'][0]['name'] if music.get('artists') else 'Unknown',
                                'album': music.get('album', {}).get('name', 'Unknown'),
                                'duration': music.get('duration_ms', 0) // 1000,
                                'acrid': music.get('acrid'),
                            }
                        }
            elif result.get('code') == 1:
                return {'copyrighted': False}
            else:
                error_msg = result.get('message', 'Unknown error')
                error_code = result.get('code', 'N/A')
                return {'copyrighted': None, 'error': f'API Error ({error_code}): {error_msg}'}
        
        return {'copyrighted': None, 'error': f'Invalid response: {result}'}
    def identify_with_timeline(self, audio_file_path,
                           chunk_seconds=10,
                           overlap_seconds=2):

        import math
        import subprocess
        import tempfile

        try:
            # Get audio duration using ffprobe
            cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                audio_file_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            total_duration = float(result.stdout.strip())

            print(f"🎧 Total duration: {total_duration}s")

            segments = []
            step = chunk_seconds - overlap_seconds
            current = 0
            
            while current < total_duration:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    chunk_path = tmp.name

                # extract chunk
                cmd = [
                    "ffmpeg",
                    "-i", audio_file_path,
                    "-ss", str(current),
                    "-t", str(chunk_seconds),
                    "-acodec", "libmp3lame",
                    chunk_path,
                    "-y"
                ]
                subprocess.run(cmd, capture_output=True)

                # run identification
                result = self.identify(chunk_path, skip_trim=True)

                if result.get("copyrighted"):
                    segments.append({
                        "start": current,
                        "end": current + chunk_seconds,
                        "music": result.get("music")
                    })
                print(f"Chunk {current}s detected:")
                print(result.get("music", {}).get("title"))
                print("ACRID:", result.get("music", {}).get("acrid"))
                os.unlink(chunk_path)
                current += step

            merged = self._merge_segments(segments)

            return {
                "copyrighted": len(merged) > 0,
                "segments": merged
            }

        except Exception as e:
                return {"copyrighted": None, "error": str(e)}
    def _merge_segments(self, segments, gap_threshold=1):

        if not segments:
            return []

        segments.sort(key=lambda x: x["start"])
        merged = [segments[0]]

        for seg in segments[1:]:
            last = merged[-1]

            last_acrid = last["music"].get("acrid")
            current_acrid = seg["music"].get("acrid")

            same_song = last_acrid == current_acrid
            close_in_time = (seg["start"] - last["end"]) <= gap_threshold

            if same_song and close_in_time:
                last["end"] = seg["end"]
            else:
                merged.append(seg)

        for seg in merged:
            seg["duration"] = round(seg["end"] - seg["start"], 2)

        return merged