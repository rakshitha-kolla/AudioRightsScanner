import numpy as np
import csv
import os

class YAMNetDetector:

    MUSIC_KEYWORDS = [
        'music', 'song', 'singing', 'choir', 'beat',
        'drum', 'guitar', 'piano', 'violin', 'flute',
        'instrument', 'orchestra', 'melody', 'rhythm',
        'pop music', 'rock music', 'hip hop', 'jazz',
        'electronic music', 'background music', 'soundtrack'
    ]

    def __init__(self,
                 confidence_threshold=0.1,       
                 background_music_threshold=0.05, 
                 min_segment_duration=2.0,
                 merge_gap=2.0):
        self.confidence_threshold        = confidence_threshold
        self.background_music_threshold  = background_music_threshold
        self.min_segment_duration        = min_segment_duration
        self.merge_gap                   = merge_gap
        self.model       = None
        self.class_names = []
        self.music_class_ids = set()

    def load_model(self):
        if self.model is not None:
            return
        import tensorflow_hub as hub
        import tensorflow as tf

        print("Loading YAMNet model...")
        self.model = hub.load("https://tfhub.dev/google/yamnet/1")

        class_map_path = self.model.class_map_path().numpy().decode('utf-8')
        with tf.io.gfile.GFile(class_map_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.class_names.append(row['display_name'])

        self.music_class_ids = {
            i for i, name in enumerate(self.class_names)
            if any(kw in name.lower() for kw in self.MUSIC_KEYWORDS)
        }
        print(f"YAMNet loaded — {len(self.class_names)} classes, {len(self.music_class_ids)} music-related")


    def _load_audio_16k(self, audio_path):
        import librosa
        print(f"Loading audio: {audio_path}")
        waveform, _ = librosa.load(audio_path, sr=16000, mono=True)
        return waveform.astype(np.float32)

    def detect_music_frames(self, waveform):
        import tensorflow as tf
        self.load_model()

        scores, _, _ = self.model(tf.constant(waveform, dtype=tf.float32))
        scores = scores.numpy()

        FRAME_DURATION = 0.96
        FRAME_HOP      = 0.48

        music_frames    = []
        top_classes_seen = {}
        second_debug     = {}

        for frame_idx, frame_scores in enumerate(scores):
            top_id   = int(np.argmax(frame_scores))
            top_name = self.class_names[top_id] if top_id < len(self.class_names) else "unknown"
            top_classes_seen[top_name] = top_classes_seen.get(top_name, 0) + 1

            music_score = max(
                (float(frame_scores[cid]) for cid in self.music_class_ids if cid < len(frame_scores)),
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

        print(f"\n Per-second breakdown (threshold={self.confidence_threshold}, bgm_threshold={self.background_music_threshold}):")
        print(f"   {'sec':>4}  {'score':>6}  {'status':<18}  top_class")
        print(f"   {'─'*4}  {'─'*6}  {'─'*18}  {'─'*25}")
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

        print(f"\nTop classes: " + ", ".join(
            f"{n}({c})" for n, c in sorted(top_classes_seen.items(), key=lambda x: -x[1])[:6]
        ))
        print(f"Music frames: {len(music_frames)} / {len(scores)}")
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

        duration = seg_end - seg_start
        y, sr = librosa.load(audio_path, sr=22050, mono=True,
                             offset=seg_start, duration=duration)

        if len(y) < sr * 2:
            return []

        hop = 512

        chroma      = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
        chroma_diff = np.mean(np.abs(np.diff(chroma, axis=1)), axis=0)

        mfcc      = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop)
        mfcc_diff = np.mean(np.abs(np.diff(mfcc, axis=1)), axis=0)

        def normalize(x):
            return x / x.max() if x.max() > 0 else x

        combined = (normalize(chroma_diff) + normalize(mfcc_diff)) / 2.0

        hop_duration = hop / sr
        boundaries   = []
        last_b       = -min_gap_between_boundaries

        for frame_idx, change in enumerate(combined):
            if change >= chroma_threshold:
                abs_time = seg_start + (frame_idx * hop_duration)
                if abs_time - last_b >= min_gap_between_boundaries:
                    boundaries.append(round(abs_time, 2))
                    last_b = abs_time

        if boundaries:
            print(f"{len(boundaries)} candidate boundaries: {boundaries}")
        else:
            print(f"No boundaries found (segment may be one continuous BGM)")

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

        waveform       = self._load_audio_16k(audio_path)
        total_duration = len(waveform) / 16000
        print(f"Total duration: {total_duration:.1f}s")

        music_frames = self.detect_music_frames(waveform)
        if not music_frames:
            print("No music detected. Try lowering thresholds.")
            return []

        coarse_segments = self._merge_frames_to_segments(music_frames)
        if not coarse_segments:
            return []

        final_segments = []
        for seg_start, seg_end in coarse_segments:
            print(f"\nSegment: {seg_start:.1f}s → {seg_end:.1f}s  ({seg_end-seg_start:.1f}s)")
            boundaries = self.find_boundaries_in_segment(audio_path, seg_start, seg_end)
            sub_segs   = self.split_segment_at_boundaries(seg_start, seg_end, boundaries)
            for start, end in sub_segs:
                final_segments.append({"start": round(start, 2), "end": round(end, 2)})

        print(f"\n Final segments to query ACRCloud: {len(final_segments)}")
        for i, seg in enumerate(final_segments):
            print(f"[{i+1}] {seg['start']}s → {seg['end']}s  ({seg['end']-seg['start']:.1f}s)")

        return final_segments