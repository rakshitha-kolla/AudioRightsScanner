import requests
import base64
import hmac
import hashlib
from datetime import datetime
import subprocess
import os
import tempfile


class ACRCloudService:

    def __init__(self, access_key, access_secret, host='identify-us.acrcloud.com'):
        self.access_key = access_key
        self.access_secret = access_secret
        self.host = host

    def _trim_audio(self, audio_file_path, duration_seconds=15):
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp:
                tmp_path = tmp.name
            cmd = ['ffmpeg', '-i', audio_file_path, '-t', str(duration_seconds),
                   '-q:a', '9', '-acodec', 'libmp3lame', tmp_path, '-y']
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg error: {result.stderr.decode()}")
            with open(tmp_path, 'rb') as f:
                return f.read()
        except Exception:
            with open(audio_file_path, 'rb') as f:
                return f.read()
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _extract_chunk(self, audio_file_path, start_sec, end_sec):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp:
            chunk_path = tmp.name
        cmd = ['ffmpeg', '-i', audio_file_path,
               '-ss', str(start_sec), '-t', str(end_sec - start_sec),
               '-acodec', 'libmp3lame', '-q:a', '9', chunk_path, '-y']
        subprocess.run(cmd, capture_output=True)
        return chunk_path

    def _get_probe_points(self, seg_start, seg_end, probe_interval=8.0):
        points = []
        current = seg_start
        while current < seg_end - 2.0:
            points.append(current)
            current += probe_interval
        return points

    def identify(self, audio_file_path):
        try:
            audio_data = self._trim_audio(audio_file_path, duration_seconds=15)
            timestamp = str(int(datetime.now().timestamp()))
            signature_string = f"POST\n/v1/identify\n{self.access_key}\naudio\n1\n{timestamp}"
            signature = base64.b64encode(
                hmac.new(self.access_secret.encode(), signature_string.encode(), hashlib.sha1).digest()
            ).decode()
            files = {
                'sample': audio_data,
                'access_key': (None, self.access_key),
                'timestamp': (None, timestamp),
                'signature': (None, signature),
                'data_type': (None, 'audio'),
                'signature_version': (None, '1'),
            }
            response = requests.post(f'https://{self.host}/v1/identify', files=files, timeout=30)
            return self._parse_result(response.json())
        except FileNotFoundError:
            return {'copyrighted': None, 'error': 'File not found'}
        except requests.exceptions.RequestException as e:
            return {'copyrighted': None, 'error': f'Network error: {str(e)}'}
        except Exception as e:
            return {'copyrighted': None, 'error': f'Error: {str(e)}'}

    def _parse_result(self, result):
        status_block = result.get('status') or result
        code = status_block.get('code')
        if code == 0:
            music_list = result.get('metadata', {}).get('music', [])
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
        elif code == 1:
            return {'copyrighted': False}
        else:
            msg = status_block.get('msg') or result.get('message', 'Unknown error')
            return {'copyrighted': None, 'error': f'API Error ({code}): {msg}'}
        return {'copyrighted': None, 'error': f'Invalid response: {result}'}

    def identify_with_yamnet(self, audio_file_path,
                              confidence_threshold=0.1,
                              min_segment_duration=3.0,
                              chroma_threshold=0.35,
                              detector_instance=None):
        from .yamnet_detector import YAMNetDetector
        try:
            if detector_instance is not None:
                detector = detector_instance.clone()
                detector.confidence_threshold = confidence_threshold
                detector.min_segment_duration = min_segment_duration
            else:
                detector = YAMNetDetector(
                    confidence_threshold=confidence_threshold,
                    min_segment_duration=min_segment_duration
                )

            candidate_segments = detector.get_music_segments(audio_file_path)
            if not candidate_segments:
                return {"copyrighted": False, "segments": []}

            confirmed_segments = []
            for seg in candidate_segments:
                start = seg["start"]
                end = seg["end"]
                if end - start < 2.0:
                    continue
                for probe_start in self._get_probe_points(start, end, probe_interval=8.0):
                    probe_end = min(probe_start + 12.0, end)
                    chunk_path = self._extract_chunk(audio_file_path, probe_start, probe_end)
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
                            confirmed_segments[-1]["end"] = round(probe_end, 2)
                            confirmed_segments[-1]["duration"] = round(probe_end - confirmed_segments[-1]["start"], 2)
                        else:
                            confirmed_segments.append({
                                "start": round(probe_start, 2),
                                "end": round(probe_end, 2),
                                "duration": round(probe_end - probe_start, 2),
                                "music": music
                            })

            return {"copyrighted": len(confirmed_segments) > 0, "segments": confirmed_segments}
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"copyrighted": None, "error": str(e)}

    def identify_with_timeline(self, audio_file_path, chunk_seconds=10, overlap_seconds=2):
        try:
            cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                   "-of", "default=noprint_wrappers=1:nokey=1", audio_file_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            total_duration = float(result.stdout.strip())
            segments = []
            step = chunk_seconds - overlap_seconds
            current = 0
            while current < total_duration:
                chunk_path = self._extract_chunk(audio_file_path, current, current + chunk_seconds)
                result = self.identify(chunk_path)
                os.unlink(chunk_path)
                if result.get("copyrighted"):
                    segments.append({"start": current, "end": current + chunk_seconds, "music": result.get("music")})
                current += step
            merged = self._merge_segments(segments)
            return {"copyrighted": len(merged) > 0, "segments": merged}
        except Exception as e:
            return {"copyrighted": None, "error": str(e)}

    def merge_overlapping_segments(segments):
        if not segments:
            return []

        segments.sort(key=lambda x: x["start"])
        merged = [segments[0]]

        for current in segments[1:]:
            last = merged[-1]

            if current["start"] <= last["end"]:
                if current["duration"] > last["duration"]:
                    merged[-1] = current
            else:
                merged.append(current)

        return merged