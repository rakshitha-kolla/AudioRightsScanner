import numpy as np
import csv
import os

TFLITE_INPUT_LENGTH = 15600
FRAME_DURATION = 0.975
FRAME_HOP      = 0.4875  

MUSIC_KEYWORDS = [
    'music', 'song', 'singing', 'choir', 'beat',
    'drum', 'guitar', 'piano', 'violin', 'flute',
    'instrument', 'orchestra', 'melody', 'rhythm',
    'pop music', 'rock music', 'hip hop', 'jazz',
    'electronic music', 'background music', 'soundtrack'
]

class YAMNetDetector:

    def __init__(self,
                 model_path='yamnet.tflite',
                 class_map_path='yamnet_class_map.csv',
                 confidence_threshold=0.1,
                 background_music_threshold=0.05,
                 min_segment_duration=2.0,
                 merge_gap=2.0):

        self.model_path              = model_path
        self.class_map_path          = class_map_path
        self.confidence_threshold    = confidence_threshold
        self.background_music_threshold = background_music_threshold
        self.min_segment_duration    = min_segment_duration
        self.merge_gap               = merge_gap
        self.interpreter             = None
        self.class_names             = []
        self.music_class_ids         = set()


    def load_model(self):
        if self.interpreter is not None:
            return

        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            import tensorflow.lite as tflite

        print(f"Loading YAMNet TFLite model from {self.model_path}...")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"YAMNet model not found at '{self.model_path}'.\n"
                "Download it from: https://huggingface.co/thelou1s/yamnet/resolve/main/lite-model_yamnet_classification_tflite_1.tflite\n"
                "and save as 'yamnet.tflite' in your project root."
            )

        self.interpreter = tflite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()

        if not os.path.exists(self.class_map_path):
            raise FileNotFoundError(
                f"Class map not found at '{self.class_map_path}'.\n"
                "Download it from: https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv\n"
                "and save as 'yamnet_class_map.csv' in your project root."
            )

        with open(self.class_map_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.class_names.append(row['display_name'])

        self.music_class_ids = {
            i for i, name in enumerate(self.class_names)
            if any(kw in name.lower() for kw in MUSIC_KEYWORDS)
        }

        print(f"YAMNet TFLite ready — {len(self.class_names)} classes, {len(self.music_class_ids)} music-related")

    def _load_audio_16k(self, audio_path):
        import librosa
        waveform, _ = librosa.load(audio_path, sr=16000, mono=True)
        return waveform.astype(np.float32)

    def _run_inference(self, chunk):
        input_details  = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        if len(chunk) < TFLITE_INPUT_LENGTH:
            chunk = np.pad(chunk, (0, TFLITE_INPUT_LENGTH - len(chunk)))
        else:
            chunk = chunk[:TFLITE_INPUT_LENGTH]

        self.interpreter.set_tensor(input_details[0]['index'], chunk.reshape(1, -1))
        self.interpreter.invoke()
        scores = self.interpreter.get_tensor(output_details[0]['index'])[0]  # shape [521]
        return scores

    def detect_music_frames(self, waveform):
        self.load_model()

        total_samples = len(waveform)
        hop_samples   = int(FRAME_HOP * 16000)

        music_frames     = []
        top_classes_seen = {}
        second_debug     = {}

        frame_idx = 0
        sample_pos = 0

        while sample_pos + TFLITE_INPUT_LENGTH <= total_samples:
            chunk  = waveform[sample_pos: sample_pos + TFLITE_INPUT_LENGTH]
            scores = self._run_inference(chunk)

            top_id   = int(np.argmax(scores))
            top_name = self.class_names[top_id] if top_id < len(self.class_names) else "unknown"
            top_classes_seen[top_name] = top_classes_seen.get(top_name, 0) + 1

            music_score = max(
                (float(scores[cid]) for cid in self.music_class_ids if cid < len(scores)),
                default=0.0
            )

            sec = int(frame_idx * FRAME_HOP)
            if sec not in second_debug or music_score > second_debug[sec][0]:
                second_debug[sec] = (music_score, top_name)

            is_clear_music      = music_score >= self.confidence_threshold
            is_bgm_under_speech = (
                'speech' in top_name.lower() and
                music_score >= self.background_music_threshold
            )

            if is_clear_music or is_bgm_under_speech:
                start = frame_idx * FRAME_HOP
                music_frames.append((start, start + FRAME_DURATION, music_score))

            frame_idx  += 1
            sample_pos += hop_samples

        # Per-second summary
        print(f"\n Per-second breakdown:")
        print(f"   {'sec':>4}  {'score':>6}  {'status':<18}  top_class")
        for sec in sorted(second_debug.keys()):
            score, top_cls = second_debug[sec]
            if score >= self.confidence_threshold:
                status = "✓ MUSIC"
            elif 'speech' in top_cls.lower() and score >= self.background_music_threshold:
                status = "✓ BGM+SPEECH"
            else:
                status = "─"
            bar = "█" * int(score * 30)
            print(f"   {sec:>4}s  {score:.3f}  {status:<18}  {top_cls:<25} {bar}")

        print(f"\n Top classes: " + ", ".join(
            f"{n}({c})" for n, c in sorted(top_classes_seen.items(), key=lambda x: -x[1])[:6]
        ))
        print(f" Music frames: {len(music_frames)} / {frame_idx}")
        return music_frames

    def _merge_frames_to_segments(self, music_frames):
        if not music_frames:
            return []

        segments = []
        cur_start, cur_end, _ = music_frames[0]

        for start, end, score in music_frames[1:]:
            if start - cur_end <= self.merge_gap:
                cur_end = max(cur_end, end)
            else:
                segments.append((cur_start, cur_end))
                cur_start, cur_end = start, end
        segments.append((cur_start, cur_end))

        before   = len(segments)
        segments = [(s, e) for s, e in segments if (e - s) >= self.min_segment_duration]
        print(f"Merged into {len(segments)} segments (dropped {before - len(segments)} short ones)")
        return segments

    def find_boundaries_in_segment(self, audio_path, seg_start, seg_end,
                                    chroma_threshold=0.3,
                                    min_gap_between_boundaries=5.0):
        import librosa

        y, sr = librosa.load(audio_path, sr=22050, mono=True,
                             offset=seg_start, duration=seg_end - seg_start)
        if len(y) < sr * 2:
            return []

        hop = 512
        chroma    = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
        mfcc      = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop)

        def norm_diff(feat):
            d = np.mean(np.abs(np.diff(feat, axis=1)), axis=0)
            return d / d.max() if d.max() > 0 else d

        combined     = (norm_diff(chroma) + norm_diff(mfcc)) / 2.0
        hop_duration = hop / sr
        boundaries   = []
        last_b       = -min_gap_between_boundaries

        for i, change in enumerate(combined):
            if change >= chroma_threshold:
                abs_time = seg_start + (i * hop_duration)
                if abs_time - last_b >= min_gap_between_boundaries:
                    boundaries.append(round(abs_time, 2))
                    last_b = abs_time

        if boundaries:
            print(f" {len(boundaries)} boundaries: {boundaries}")
        return boundaries

    def split_segment_at_boundaries(self, seg_start, seg_end, boundaries):
        valid = [b for b in boundaries if seg_start < b < seg_end]
        if not valid:
            return [(seg_start, seg_end)]
        points = [seg_start] + sorted(valid) + [seg_end]
        return [(points[i], points[i+1]) for i in range(len(points) - 1)]

    def get_music_segments(self, audio_path):
        print(f"\n{'='*50}")
        print(f" Analyzing: {audio_path}")
        print(f"{'='*50}")

        waveform = self._load_audio_16k(audio_path)
        print(f" Total duration: {len(waveform)/16000:.1f}s")

        music_frames = self.detect_music_frames(waveform)
        if not music_frames:
            return []

        coarse = self._merge_frames_to_segments(music_frames)
        if not coarse:
            return []

        final = []
        for seg_start, seg_end in coarse:
            print(f"\n {seg_start:.1f}s → {seg_end:.1f}s")
            boundaries = self.find_boundaries_in_segment(audio_path, seg_start, seg_end)
            for start, end in self.split_segment_at_boundaries(seg_start, seg_end, boundaries):
                final.append({"start": round(start, 2), "end": round(end, 2)})

        print(f"\n Final segments: {len(final)}")
        for i, s in enumerate(final):
            print(f"   [{i+1}] {s['start']}s → {s['end']}s  ({s['end']-s['start']:.1f}s)")
        return final