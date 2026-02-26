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
    def __init__(self, access_key, access_secret, host='identify-us.acrcloud.com'):
        self.access_key = access_key
        self.access_secret = access_secret
        self.host = host
    
    def _trim_audio(self, audio_file_path, duration_seconds=15):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                tmp_path = tmp.name
            
            cmd = [
                'ffmpeg',
                '-i', audio_file_path,
                '-t', str(duration_seconds),
                '-q:a', '9',
                '-acodec', 'libmp3lame',
                tmp_path,
                '-y'
            ]
            
            subprocess.run(cmd, capture_output=True, check=True)
            
            with open(tmp_path, 'rb') as f:
                trimmed_data = f.read()
            
            os.unlink(tmp_path)
            return trimmed_data
        
        except FileNotFoundError:
            print("  ffmpeg not installed. Using original file (may fail if too large)")
            with open(audio_file_path, 'rb') as f:
                return f.read()
        except Exception as e:
            print(f"  Could not trim audio: {e}. Using original file")
            with open(audio_file_path, 'rb') as f:
                return f.read()

    def _extract_chunk(self, audio_file_path, start_sec, end_sec):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            chunk_path = tmp.name

        duration = end_sec - start_sec
        cmd = [
            'ffmpeg',
            '-i', audio_file_path,
            '-ss', str(start_sec),
            '-t',  str(duration),
            '-acodec', 'libmp3lame',
            chunk_path,
            '-y'
        ]
        subprocess.run(cmd, capture_output=True)
        return chunk_path
    
    def identify(self, audio_file_path):
        try:
            print("  Trimming audio to 15 seconds...")
            audio_data = self._trim_audio(audio_file_path, duration_seconds=15)
            
            print(f" Trimmed file size: {len(audio_data) / 1024 / 1024:.2f} MB")
            timestamp = str(int(datetime.now().timestamp()))
            signature_string = f"POST\n/v1/identify\n{self.access_key}\naudio\n1\n{timestamp}"
            
            signature = base64.b64encode(
                hmac.new(
                    self.access_secret.encode(),
                    signature_string.encode(),
                    hashlib.sha1
                ).digest()
            ).decode()
            
            files = {
                'sample': audio_data,
                'access_key': (None, self.access_key),
                'timestamp': (None, timestamp),
                'signature': (None, signature),
                'data_type': (None, 'audio'),
                'signature_version': (None, '1'),
            }
            
            url = f'https://{self.host}/v1/identify'
            print(f" Sending to ACRCloud...")
            
            response = requests.post(url, files=files, timeout=30)
            print(f" Response Status: {response.status_code}")
            
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

    def _get_probe_points(self, seg_start, seg_end, probe_interval=8.0):
        """
        Generate probe timestamps within a segment.
        For a 19s segment (74s→93s) with interval=8:
            probes = [74, 82, 90]  — catches BGM change at 85s between probe 82 and 90
        """
        points = []
        current = seg_start
        while current < seg_end - 2.0:  # stop if less than 2s remains
            points.append(current)
            current += probe_interval
        return points

    def identify_with_yamnet(self, audio_file_path,
                              confidence_threshold=0.1,
                              min_segment_duration=3.0,
                              chroma_threshold=0.35):
        
        from yamnet_detector import YAMNetDetector

        try:
            detector = YAMNetDetector(
                confidence_threshold=confidence_threshold,
                min_segment_duration=min_segment_duration
            )

            print("\n Running YAMNet + Chroma analysis...")
            candidate_segments = detector.get_music_segments(audio_file_path)

            if not candidate_segments:
                print("No music segments found by YAMNet")
                return {"copyrighted": False, "segments": []}

            print(f"\n Sending {len(candidate_segments)} segments to ACRCloud...")

            confirmed_segments = []
            seen_acrids = {}  

            for i, seg in enumerate(candidate_segments):
                start = seg["start"]
                end   = seg["end"]

                print(f"\n[{i+1}/{len(candidate_segments)}] Querying {start}s → {end}s")
                probe_points = self._get_probe_points(start, end, probe_interval=8.0)
                print(f"   Probing at {len(probe_points)} points: {[round(p,1) for p in probe_points]}")

                for probe_start in probe_points:
                    probe_end   = min(probe_start + 12.0, end)  # 12s probe window
                    chunk_path  = self._extract_chunk(audio_file_path, probe_start, probe_end)

                    try:
                        result = self.identify(chunk_path)
                    finally:
                        if os.path.exists(chunk_path):
                            os.unlink(chunk_path)

                    if result.get("copyrighted"):
                        music = result["music"]
                        acrid = music.get("acrid")
                        prev_acrid = confirmed_segments[-1].get("music", {}).get("acrid") if confirmed_segments else None

                        if confirmed_segments and acrid and acrid == prev_acrid:
                            confirmed_segments[-1]["end"]      = round(probe_end, 2)
                            confirmed_segments[-1]["duration"] = round(probe_end - confirmed_segments[-1]["start"], 2)
                            print(f"    {probe_start:.1f}s: same song, extending ({music.get('title')})")
                        else:
                            if prev_acrid and acrid != prev_acrid:
                                print(f"    {probe_start:.1f}s: acrid changed → new song!")
                            confirmed_segments.append({
                                "start":    round(probe_start, 2),
                                "end":      round(probe_end,   2),
                                "duration": round(probe_end - probe_start, 2),
                                "music":    music
                            })
                            print(f"    {probe_start:.1f}s: {music.get('title')} — {music.get('artist')}")
                    else:
                        print(f"    {probe_start:.1f}s: no match")

            return {
                "copyrighted": len(confirmed_segments) > 0,
                "segments":    confirmed_segments
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"copyrighted": None, "error": str(e)}

    def identify_with_timeline(self, audio_file_path,
                               chunk_seconds=10,
                               overlap_seconds=2):
        try:
            cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                audio_file_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            total_duration = float(result.stdout.strip())

            print(f"Total duration: {total_duration}s")

            segments = []
            step = chunk_seconds - overlap_seconds
            current = 0

            while current < total_duration:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    chunk_path = tmp.name

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

                result = self.identify(chunk_path)

                if result.get("copyrighted"):
                    segments.append({
                        "start": current,
                        "end":   current + chunk_seconds,
                        "music": result.get("music")
                    })

                os.unlink(chunk_path)
                current += step

            merged = self._merge_segments(segments)

            return {
                "copyrighted": len(merged) > 0,
                "segments":    merged
            }

        except Exception as e:
            return {"copyrighted": None, "error": str(e)}

    def _merge_segments(self, segments, gap_threshold=3):
        if not segments:
            return []

        segments.sort(key=lambda x: x["start"])
        merged = [segments[0]]

        for seg in segments[1:]:
            last = merged[-1]
            if seg["start"] - last["end"] <= gap_threshold:
                last["end"] = seg["end"]
            else:
                merged.append(seg)

        for seg in merged:
            seg["duration"] = round(seg["end"] - seg["start"], 2)

        return merged