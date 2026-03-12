"""
Binaural inference clip synthesis for BinauralSSLNet.

Generates a small number of long (~1 min) binaural WAV files from real sound
effects convolved with HRTF databases, together with per-frame ground-truth
heatmaps that match the sliding-window schedule used by inference.py.

Each clip is divided into random-duration segments (5–15 s each), with
independently sampled source positions per segment.  For every sliding window
the actual RMS energy of each source is measured from that window's audio,
then expressed as a dB value relative to the loudest source in the window
(mapped to the 0–30 dB range that generate_heatmap expects).  This means
heatmap sigma and amplitude both vary naturally with the audio content — even
within a fixed-position segment.

The per-frame heatmap array has shape [N_frames, 72, 37] and is stored as a
single .npy file alongside the audio file.

Usage:
    python synthesize_inference_data.py [--n_clips 10] [--duration_sec 60]
                                        [--out_dir ./data/inference]
                                        [--window_ms 128] [--step_ms 64]
                                        [--seg_min 5] [--seg_max 15]
"""

import argparse
import json
import random
import warnings
from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

from utils.audio_processing import SAMPLE_RATE, FEATURE_SR
from utils.hrtf_synthesis import HRTFDatabasePool
from utils.heatmap_generator import generate_heatmap, SILENCE_DB
from data_generation import (
    AudioCache,
    sample_n_sources,
    sample_source_positions,
    sample_db,
    BUFFER_SAMPLES,
    compute_rms_db,
)

warnings.filterwarnings('ignore')

BASE_DIR          = Path(__file__).parent
SOUND_EFFECTS_DIR = BASE_DIR / 'data' / 'sound_effects'
HRIR_DIR          = BASE_DIR / 'data' / 'hrir'

# The loudest source in each window is mapped to this reference dB before
# calling generate_heatmap.  Matches the upper end of the training dB range.
REFERENCE_DB = 25.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def rms(x: np.ndarray) -> float:
    """Linear RMS of a 1-D array."""
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-12))


def sources_to_effective_db(
    source_rms: List[Tuple[float, float, float]],
) -> List[Tuple[float, float, float]]:
    """
    Convert per-source linear RMS values (az, el, rms_linear) to dB values
    that fall in the range generate_heatmap expects (0–30 dB), using the
    loudest source in this window as a reference.

    Returns [(az, el, effective_db), ...], omitting sources that would fall
    below SILENCE_DB.
    """
    if not source_rms:
        return []

    rms_max = max(r for _, _, r in source_rms)
    if rms_max < 1e-10:
        return []

    result = []
    for az, el, r in source_rms:
        if r < 1e-10:
            continue
        # dB relative to the loudest source in this window
        db_rel = 20.0 * np.log10(r / rms_max + 1e-12)
        effective_db = REFERENCE_DB + db_rel
        if effective_db > SILENCE_DB:
            result.append((az, el, float(effective_db)))

    return result


# ── Segment planning ──────────────────────────────────────────────────────────

def make_segment_plan(
    duration_sec: float,
    seg_min: float = 5.0,
    seg_max: float = 15.0,
) -> List[Tuple[float, float, List[Tuple[float, float, float]]]]:
    """
    Divide [0, duration_sec] into random-length segments.
    Returns list of (start_sec, end_sec, [(az, el, nominal_db), ...]).
    """
    segments = []
    t = 0.0
    while t < duration_sec:
        remaining = duration_sec - t
        seg_dur = min(random.uniform(seg_min, seg_max), remaining)
        if seg_dur < 1.0:
            if segments:
                s, _, srcs = segments[-1]
                segments[-1] = (s, duration_sec, srcs)
            break
        n_src   = sample_n_sources()
        positions = sample_source_positions(n_src)
        sources = [(az, el, sample_db()) for az, el in positions]
        segments.append((t, t + seg_dur, sources))
        t += seg_dur
    return segments


# ── Synthesis ─────────────────────────────────────────────────────────────────

def synthesize_clip(
    audio_cache: AudioCache,
    hrtf_pool,
    duration_sec: float,
    window_ms: int = 128,
    step_ms: int = 64,
    seg_min: float = 5.0,
    seg_max: float = 15.0,
) -> Tuple[np.ndarray, np.ndarray, List]:
    """
    Synthesize one long binaural clip.

    For each sliding window the actual RMS energy of each source is measured
    from the corresponding slice of the per-source binaural track, then
    mapped to an effective dB that drives generate_heatmap.

    Returns:
        audio:          [2, n_samples] float32 at FEATURE_SR
        frame_heatmaps: [N_frames, 72, 37] float32
        segments_meta:  list of dicts (for metadata.json)
    """
    segments = make_segment_plan(duration_sec, seg_min, seg_max)
    hrtf_db  = hrtf_pool.get_random()

    # ── Phase 1: Synthesise each segment, keep individual source signals ──────
    # seg_data: [(mix [2, n_synth], [(az, el, src [2, n_synth]), ...]), ...]
    seg_data = []

    for seg_start, seg_end, seg_sources in segments:
        seg_dur = seg_end - seg_start
        n_synth = int(seg_dur * SAMPLE_RATE)
        total   = n_synth + BUFFER_SAMPLES

        seg_mix     = np.zeros((2, total), dtype=np.float32)
        src_signals = []   # [(az, el, binaural [2, n_synth]), ...]

        for az, el, db in seg_sources:
            mono     = audio_cache.get_random_segment(total)
            binaural = hrtf_db.synthesize(mono, az, el)
            cur_db   = compute_rms_db(binaural[0])
            if cur_db > -70:
                scale    = 10.0 ** ((db - cur_db) / 20.0)
                binaural = binaural * scale
            seg_mix += binaural
            # Strip the HRTF buffer so indices align with seg_mix
            src_signals.append((az, el, binaural[:, BUFFER_SAMPLES:]))

        seg_data.append((seg_mix[:, BUFFER_SAMPLES:], src_signals))

    # ── Phase 2: Concatenate and peak-normalise ───────────────────────────────
    binaural_mix = np.concatenate([sd[0] for sd in seg_data], axis=1)
    peak = np.max(np.abs(binaural_mix))
    norm = peak if peak > 1e-6 else 1.0
    binaural_mix /= norm

    # ── Phase 3: Downsample mix to FEATURE_SR ────────────────────────────────
    audio = np.stack([
        librosa.resample(binaural_mix[0], orig_sr=SAMPLE_RATE, target_sr=FEATURE_SR),
        librosa.resample(binaural_mix[1], orig_sr=SAMPLE_RATE, target_sr=FEATURE_SR),
    ], axis=0).astype(np.float32)
    del binaural_mix

    expected = int(duration_sec * FEATURE_SR)
    if audio.shape[1] > expected:
        audio = audio[:, :expected]
    elif audio.shape[1] < expected:
        audio = np.pad(audio, ((0, 0), (0, expected - audio.shape[1])))

    # ── Phase 4: Downsample individual source signals to FEATURE_SR ──────────
    # seg_tracks_16k: [(start_sample, n_samples, [(az, el, src_16k), ...]), ...]
    seg_tracks_16k = []
    t_sample_16k   = 0

    for (_, src_signals_44k), _ in zip(seg_data, segments):
        src_tracks = []
        n_seg_16k  = None

        for az, el, binaural_44k in src_signals_44k:
            # Apply the same global normalization factor
            src_l = librosa.resample(
                binaural_44k[0] / norm, orig_sr=SAMPLE_RATE, target_sr=FEATURE_SR)
            src_r = librosa.resample(
                binaural_44k[1] / norm, orig_sr=SAMPLE_RATE, target_sr=FEATURE_SR)
            src_16k = np.stack([src_l, src_r], axis=0).astype(np.float32)
            src_tracks.append((az, el, src_16k))
            if n_seg_16k is None:
                n_seg_16k = src_16k.shape[1]

        if n_seg_16k is None:
            # Segment had no sources (shouldn't happen, but be safe)
            n_seg_16k = int((_ [1] - _[0]) * FEATURE_SR)

        seg_tracks_16k.append((t_sample_16k, n_seg_16k, src_tracks))
        t_sample_16k += n_seg_16k

    del seg_data  # free 44100 Hz memory

    # ── Phase 5: Per-frame heatmaps ───────────────────────────────────────────
    window_samples = int(window_ms / 1000 * FEATURE_SR)
    step_samples   = int(step_ms   / 1000 * FEATURE_SR)
    n_samples      = audio.shape[1]

    def frame_heatmap(win_start: int, win_len: int) -> np.ndarray:
        """
        Compute the GT heatmap for one window.

        1. Find the segment whose time range contains the window centre.
        2. For each source in that segment, measure the actual RMS of its
           binaural signal in the slice [win_start, win_start+win_len].
        3. Map all RMS values to effective dBs relative to the loudest source,
           shifted to the 0–30 dB range that generate_heatmap expects.
        4. Call generate_heatmap with the effective (az, el, db) triples.
        """
        centre_t = (win_start + win_len / 2) / FEATURE_SR

        # Locate the segment containing the window centre
        active_seg = seg_tracks_16k[-1]
        for seg in seg_tracks_16k:
            s_start_t = seg[0] / FEATURE_SR
            s_end_t   = (seg[0] + seg[1]) / FEATURE_SR
            if s_start_t <= centre_t < s_end_t:
                active_seg = seg
                break

        seg_start_s, _, src_tracks = active_seg

        # Measure RMS of each source in this specific window
        source_rms_list = []
        for az, el, src_16k in src_tracks:
            local_start = win_start - seg_start_s
            local_end   = local_start + win_len
            local_start = max(0, local_start)
            local_end   = min(src_16k.shape[1], local_end)
            if local_end <= local_start:
                continue
            r = rms(src_16k[0, local_start:local_end])
            source_rms_list.append((az, el, r))

        # Convert measured RMS to effective dBs relative to loudest source
        effective_sources = sources_to_effective_db(source_rms_list)
        if not effective_sources:
            return np.zeros((72, 37), dtype=np.float32)

        return generate_heatmap(effective_sources)

    frame_heatmaps = []
    start = 0
    while start + window_samples <= n_samples:
        frame_heatmaps.append(frame_heatmap(start, window_samples))
        start += step_samples
    if start < n_samples:
        frame_heatmaps.append(frame_heatmap(start, n_samples - start))

    frame_heatmaps = np.stack(frame_heatmaps, axis=0)  # [N_frames, 72, 37]

    segments_meta = [
        {
            'start_sec': float(s),
            'end_sec':   float(e),
            'sources': [
                {'azimuth': float(az), 'elevation': float(el), 'nominal_db': float(db)}
                for az, el, db in srcs
            ],
        }
        for s, e, srcs in segments
    ]

    return audio, frame_heatmaps, segments_meta


# ── Main loop ─────────────────────────────────────────────────────────────────

def generate_inference_clips(
    n_clips: int = 10,
    duration_sec: float = 60.0,
    out_dir: str = None,
    window_ms: int = 128,
    step_ms: int = 64,
    seg_min: float = 5.0,
    seg_max: float = 15.0,
):
    out_path     = Path(out_dir) if out_dir else BASE_DIR / 'data' / 'inference'
    audio_dir    = out_path / 'audio'
    heatmaps_dir = out_path / 'heatmaps'
    audio_dir.mkdir(parents=True, exist_ok=True)
    heatmaps_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading audio cache from {SOUND_EFFECTS_DIR} ...")
    audio_cache = AudioCache(str(SOUND_EFFECTS_DIR))
    audio_cache.load_all()

    print(f"Loading HRTF pool from {HRIR_DIR} ...")
    hrtf_pool = HRTFDatabasePool(str(HRIR_DIR))
    hrtf_pool.get(hrtf_pool.sofa_paths[0])
    print(f"  Found {hrtf_pool.n_databases} SOFA file(s).")

    n_frames_est = (int(duration_sec * FEATURE_SR) - int(window_ms / 1000 * FEATURE_SR)) \
                   // int(step_ms / 1000 * FEATURE_SR) + 1
    print(f"\nWindow: {window_ms} ms  Step: {step_ms} ms  "
          f"Estimated frames/clip: ~{n_frames_est}")
    print(f"Synthesizing {n_clips} clip(s) of ~{duration_sec:.0f} s each ...")

    metadata = []

    for i in tqdm(range(n_clips), desc='Synthesizing', unit='clip'):
        try:
            audio, frame_heatmaps, segments_meta = synthesize_clip(
                audio_cache, hrtf_pool, duration_sec,
                window_ms=window_ms, step_ms=step_ms,
                seg_min=seg_min, seg_max=seg_max,
            )

            sf.write(audio_dir    / f'{i:06d}.wav', audio.T, FEATURE_SR)
            np.save(heatmaps_dir  / f'{i:06d}.npy', frame_heatmaps)

            n_frames = frame_heatmaps.shape[0]
            entry = {
                'id': i,
                'duration_sec': duration_sec,
                'n_samples': audio.shape[1],
                'sample_rate': FEATURE_SR,
                'window_ms': window_ms,
                'step_ms': step_ms,
                'n_frames': n_frames,
                'n_segments': len(segments_meta),
                'segments': segments_meta,
            }
            metadata.append(entry)

            seg_summary = ', '.join(
                f"[{s['start_sec']:.0f}–{s['end_sec']:.0f}s "
                f"{len(s['sources'])}src]"
                for s in segments_meta
            )
            tqdm.write(f'  [{i:03d}] {n_frames} frames | {seg_summary}')

        except Exception as e:
            tqdm.write(f'\n[Warning] Clip {i} failed: {e}')
            metadata.append({'id': i, 'error': str(e)})

    meta_path = out_path / 'metadata.json'
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    valid = sum(1 for m in metadata if 'error' not in m)
    print(f"\nDone. {valid}/{n_clips} clip(s) saved → {out_path}")
    if valid:
        print(f"\nRun inference on the first clip:")
        print(f"  python inference.py --audio {audio_dir / '000000.wav'}")
        print(f"\nRun inference without popup (saves video):")
        print(f"  python inference.py --audio {audio_dir / '000000.wav'} --monitor no")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Synthesize long binaural clips for inference evaluation')
    parser.add_argument('--n_clips', type=int, default=10)
    parser.add_argument('--duration_sec', type=float, default=60.0)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--window_ms', type=int, default=128)
    parser.add_argument('--step_ms', type=int, default=64)
    parser.add_argument('--seg_min', type=float, default=5.0)
    parser.add_argument('--seg_max', type=float, default=15.0)
    args = parser.parse_args()

    generate_inference_clips(
        n_clips=args.n_clips,
        duration_sec=args.duration_sec,
        out_dir=args.out_dir,
        window_ms=args.window_ms,
        step_ms=args.step_ms,
        seg_min=args.seg_min,
        seg_max=args.seg_max,
    )
