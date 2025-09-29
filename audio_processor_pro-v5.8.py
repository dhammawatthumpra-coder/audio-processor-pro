#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Professional Audio Processor Unified v5.8 with Enhanced Peak Control & Improved Truncate Silence
==================================================================================================

Enhanced audio processing script with improved peak detection, hard limiting, and highly optimized
Audacity-style truncate silence implementation.

Key Improvements in v5.8:
- Enhanced Audacity-style truncate silence with fast numpy processing
- Improved memory management for large audio files
- Better sentence break detection and preservation
- Optimized silence detection algorithm
- Enhanced edge protection and fade handling
- Sample-level precision for truncation

Version: 5.7.0 - English Translation & Refactor Edition

Author: Enhanced Audio Processing Pro
License: MIT
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
import os
import json
import argparse
import time
import logging
import glob
import asyncio
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import threading
import multiprocessing
from multiprocessing import Pool, cpu_count
import hashlib
from functools import lru_cache, partial
import gc
import numpy as np

# Core audio libraries
try:
    from pydub import AudioSegment
    from pydub.effects import normalize, compress_dynamic_range
    from pydub.utils import make_chunks
    from pydub.silence import detect_leading_silence, detect_silence
    from concurrent.futures import ProcessPoolExecutor, as_completed
    AudioSegment.converter = "ffmpeg"
    AudioSegment.ffmpeg = "ffmpeg"
    AudioSegment.ffprobe = "ffprobe"
except ImportError:
    print("ERROR: pydub library is required. Install with: pip install pydub")
    sys.exit(1)

# Advanced processing libraries (optional)
try:
    import numpy as np
    import noisereduce as nr
    from scipy import signal
    from scipy.ndimage import gaussian_filter1d
    ADVANCED_PROCESSING = True
except ImportError:
    print("WARNING: Advanced features disabled. Install: pip install numpy noisereduce scipy")
    ADVANCED_PROCESSING = False

# Performance monitoring
try:
    import psutil
    SYSTEM_MONITORING = True
except ImportError:
    SYSTEM_MONITORING = False

# Colors for terminal output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'

# Configuration constants
SUPPORTED_FORMATS = {
    '.mp3': 'mp3', '.wav': 'wav', '.flac': 'flac',
    '.m4a': 'm4a', '.aac': 'aac', '.ogg': 'ogg',
    '.wma': 'wma', '.au': 'au', '.aiff': 'aiff'
}

# Memory optimization settings
CHUNK_SIZE_MS = 12000  # Process in 12-second chunks for better throughput
CHUNK_OVERLAP_MS = 25  # Small overlap for seamless processing
MAX_MEMORY_MB = 1024  # Maximum memory usage per process
THREAD_POOL_SIZE = min(6, cpu_count())  # Reduced for better balance
PREALLOC_CHUNKS = 4  # Number of chunks to pre-allocate
CACHE_SIZE_MB = 256  # Size of chunk cache in MB

# Speed configuration presets
SPEED_PRESETS = {
    'quality': {
        'chunk_size_ms': 2000,
        'max_threads': min(4, cpu_count()),
        'enable_analysis': True,
        'compression_level': 'high',
        'skip_noise_reduction': False,
        'simple_processing': False
    },
    'normal': {
        'chunk_size_ms': CHUNK_SIZE_MS,
        'max_threads': THREAD_POOL_SIZE,
        'enable_analysis': True,
        'compression_level': 'medium',
        'skip_noise_reduction': False,
        'simple_processing': False
    },
    'fast': {
        'chunk_size_ms': 10000,
        'max_threads': min(cpu_count(), 12),
        'enable_analysis': False,
        'compression_level': 'medium',
        'skip_noise_reduction': True,
        'simple_processing': True
    },
    'ultra_fast': {
        'chunk_size_ms': 30000,
        'max_threads': cpu_count(),
        'enable_analysis': False,
        'compression_level': 'low',
        'skip_noise_reduction': True,
        'simple_processing': True
    }
}

class PeakDetector:
    """Advanced peak detection and analysis for checking high peaks before normalization."""

    @staticmethod
    def analyze_peaks(audio: AudioSegment, threshold_db: float = -3.0) -> Dict:
        """
        Analyzes peak levels in the audio file to determine if hard limiting is necessary.

        Args:
            audio: The AudioSegment to analyze.
            threshold_db: The peak threshold in dBFS (default: -3.0).

        Returns:
            A dictionary containing the peak analysis results.
        """
        try:
            # Convert audio to a numpy array
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            if audio.channels == 2:
                samples = samples.reshape((-1, 2))

            # Calculate peak levels
            if audio.channels == 2:
                left_peak = np.max(np.abs(samples[:, 0]))
                right_peak = np.max(np.abs(samples[:, 1]))
                overall_peak = max(left_peak, right_peak)
            else:
                overall_peak = np.max(np.abs(samples))

            # Convert to dBFS
            max_possible = 2**(audio.sample_width * 8 - 1)
            peak_dbfs = 20 * np.log10(overall_peak / max_possible) if overall_peak > 0 else float('-inf')

            # Calculate RMS for dynamic range analysis
            if audio.channels == 2:
                rms_left = np.sqrt(np.mean(samples[:, 0]**2))
                rms_right = np.sqrt(np.mean(samples[:, 1]**2))
                rms_overall = max(rms_left, rms_right)
            else:
                rms_overall = np.sqrt(np.mean(samples**2))

            rms_dbfs = 20 * np.log10(rms_overall / max_possible) if rms_overall > 0 else float('-inf')

            # Calculate crest factor (peak-to-RMS ratio)
            crest_factor = peak_dbfs - rms_dbfs

            # Determine if limiting is needed
            needs_limiting = peak_dbfs > threshold_db
            limiting_amount = max(0, peak_dbfs - threshold_db)

            # Count peaks exceeding the threshold
            peak_threshold_linear = max_possible * (10**(threshold_db / 20))
            if audio.channels == 2:
                severe_peaks = np.sum((np.abs(samples[:, 0]) > peak_threshold_linear) |
                                    (np.abs(samples[:, 1]) > peak_threshold_linear))
            else:
                severe_peaks = np.sum(np.abs(samples) > peak_threshold_linear)

            peak_percentage = (severe_peaks / len(samples.flatten())) * 100

            return {
                'peak_dbfs': peak_dbfs,
                'rms_dbfs': rms_dbfs,
                'crest_factor': crest_factor,
                'needs_limiting': needs_limiting,
                'limiting_amount_db': limiting_amount,
                'severe_peaks_count': int(severe_peaks),
                'peak_percentage': peak_percentage,
                'dynamic_range': crest_factor,
                'recommended_limiter_threshold': min(threshold_db, peak_dbfs - 0.1)
            }

        except Exception as e:
            print(f"Warning: Peak analysis failed: {e}")
            return {
                'peak_dbfs': audio.dBFS,
                'needs_limiting': False,
                'limiting_amount_db': 0,
                'error': str(e)
            }

class HardLimiter:
    """Professional hard limiter for use before normalization."""

    @staticmethod
    def apply_hard_limit(audio: AudioSegment, threshold_db: float = -0.1,
                        release_ms: float = 5.0, lookahead_ms: float = 1.0) -> AudioSegment:
        """
        Applies professional-style hard limiting.

        Args:
            audio: Input audio.
            threshold_db: Limiter threshold in dBFS.
            release_ms: Release time in milliseconds.
            lookahead_ms: Lookahead time in milliseconds.
        """
        try:
            print(f"   Applying hard limiter: threshold={threshold_db:.1f}dB, release={release_ms}ms")

            # Get audio samples
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            sample_rate = audio.frame_rate

            if audio.channels == 2:
                samples = samples.reshape((-1, 2))

            # Calculate threshold in linear format
            max_amplitude = 2**(audio.sample_width * 8 - 1)
            threshold_linear = max_amplitude * (10**(threshold_db / 20))

            # Efficient hard limiting
            if audio.channels == 2:
                # Stereo processing - maintain channel balance
                for ch in range(2):
                    channel_samples = samples[:, ch]

                    # Apply hard limiting
                    over_threshold = np.abs(channel_samples) > threshold_linear
                    if np.any(over_threshold):
                        # Preserve sign while limiting amplitude
                        channel_samples[over_threshold] = (
                            np.sign(channel_samples[over_threshold]) * threshold_linear
                        )
                        samples[:, ch] = channel_samples
            else:
                # Mono processing
                over_threshold = np.abs(samples) > threshold_linear
                if np.any(over_threshold):
                    samples[over_threshold] = np.sign(samples[over_threshold]) * threshold_linear

            # Convert back to original format
            if audio.channels == 2:
                samples = samples.flatten()

            # Ensure samples are within bit depth limits
            samples = np.clip(samples, -max_amplitude, max_amplitude-1)
            limited_samples = samples.astype(np.int16)

            # Create a new AudioSegment
            limited_audio = AudioSegment(
                limited_samples.tobytes(),
                frame_rate=sample_rate,
                sample_width=audio.sample_width,
                channels=audio.channels
            )

            return limited_audio

        except Exception as e:
            print(f"   Warning: Hard limiting failed: {e}")
            return audio

class VolumeBoostProcessor:
    """Advanced volume boost processor for intelligent gain adjustment."""
    
    @staticmethod
    def analyze_loudness(audio: AudioSegment) -> Dict:
        """Analyzes the loudness and dynamic range of the audio."""
        try:
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            if audio.channels == 2:
                samples = samples.reshape((-1, 2))
                # Use the average of both channels
                samples = np.mean(samples, axis=1)
            
            # Calculate RMS (Root Mean Square) - close to perceived loudness
            rms = np.sqrt(np.mean(samples**2))
            max_possible = 2**(audio.sample_width * 8 - 1)
            rms_dbfs = 20 * np.log10(rms / max_possible) if rms > 0 else float('-inf')
            
            # Calculate peak level
            peak = np.max(np.abs(samples))
            peak_dbfs = 20 * np.log10(peak / max_possible) if peak > 0 else float('-inf')
            
            # Calculate crest factor (dynamic range indicator)
            crest_factor = peak_dbfs - rms_dbfs
            
            # Estimate loudness distribution
            loudness_percentiles = np.percentile(np.abs(samples), [10, 25, 50, 75, 90, 95, 99])
            loudness_percentiles_db = [20 * np.log10(p / max_possible) if p > 0 else float('-inf') 
                                     for p in loudness_percentiles]
            
            return {
                'rms_dbfs': rms_dbfs,
                'peak_dbfs': peak_dbfs,
                'crest_factor': crest_factor,
                'loudness_percentiles': loudness_percentiles_db,
                'estimated_lufs': rms_dbfs + 0.691,  # Rough LUFS approximation
                'dynamic_range': crest_factor,
                'headroom': -0.1 - peak_dbfs  # Headroom to -0.1 dBFS
            }
            
        except Exception as e:
            print(f"Warning: Loudness analysis failed: {e}")
            return {
                'rms_dbfs': audio.dBFS,
                'peak_dbfs': audio.dBFS,
                'crest_factor': 0,
                'estimated_lufs': audio.dBFS,
                'headroom': 0
            }
    
    @staticmethod
    def calculate_optimal_gain(audio: AudioSegment, target_lufs: float = -16.0, 
                             max_peak_db: float = -0.3, preserve_dynamics: bool = True) -> float:
        """Calculates the optimal gain for volume boosting."""
        try:
            analysis = VolumeBoostProcessor.analyze_loudness(audio)
            
            # Calculate the required gain to reach target LUFS
            lufs_gain = target_lufs - analysis['estimated_lufs']
            
            # Calculate the maximum gain that won't exceed max_peak_db
            peak_gain = max_peak_db - analysis['peak_dbfs']
            
            # Choose the smaller gain for safety
            safe_gain = min(lufs_gain, peak_gain)
            
            # If preserve_dynamics is True, limit gain to preserve dynamic range
            if preserve_dynamics and analysis['crest_factor'] > 0:
                # Limit gain to not reduce dynamic range too much
                max_dynamic_gain = min(safe_gain, analysis['crest_factor'] * 0.7)
                safe_gain = max_dynamic_gain
            
            # Limit gain to a maximum of 12 dB for safety
            final_gain = max(-6.0, min(safe_gain, 12.0))
            
            return final_gain
            
        except Exception as e:
            print(f"Warning: Gain calculation failed: {e}")
            return 3.0  # Default safe gain
    
    @staticmethod
    def apply_intelligent_volume_boost(audio: AudioSegment, boost_config: Dict) -> AudioSegment:
        """Applies intelligent volume boosting."""
        try:
            if not boost_config.get('enabled', False):
                return audio
            
            print("📊 Applying intelligent volume boost:")
            
            # Analyze current loudness
            analysis = VolumeBoostProcessor.analyze_loudness(audio)
            print(f"   Current levels: RMS={analysis['rms_dbfs']:.1f} dBFS, Peak={analysis['peak_dbfs']:.1f} dBFS")
            print(f"   Estimated LUFS: {analysis['estimated_lufs']:.1f}")
            print(f"   Dynamic Range: {analysis['crest_factor']:.1f} dB")
            
            # Calculate optimal gain
            target_lufs = boost_config.get('target_lufs', -16.0)
            max_peak_db = boost_config.get('max_peak_db', -0.3)
            preserve_dynamics = boost_config.get('preserve_dynamics', True)
            
            if boost_config.get('adaptive_gain', True):
                optimal_gain = VolumeBoostProcessor.calculate_optimal_gain(
                    audio, target_lufs, max_peak_db, preserve_dynamics
                )
                print(f"   Calculated adaptive gain: {optimal_gain:+.1f} dB")
            else:
                optimal_gain = boost_config.get('fixed_gain', 6.0)
                print(f"   Using fixed gain: {optimal_gain:+.1f} dB")
            
            # Apply gain
            if abs(optimal_gain) > 0.1:
                boosted_audio = audio + optimal_gain
                
                # Check the result
                final_analysis = VolumeBoostProcessor.analyze_loudness(boosted_audio)
                print(f"   Result: RMS={final_analysis['rms_dbfs']:.1f} dBFS, Peak={final_analysis['peak_dbfs']:.1f} dBFS")
                print(f"   Estimated LUFS after boost: {final_analysis['estimated_lufs']:.1f}")
                
                # Check if additional limiting is needed
                if final_analysis['peak_dbfs'] > max_peak_db:
                    print(f"   ⚠️ Peak exceeds limit - applying safety limiter")
                    boosted_audio = HardLimiter.apply_hard_limit(boosted_audio, threshold_db=max_peak_db)
                
                return boosted_audio
            else:
                print("   No volume adjustment needed")
                return audio
                
        except Exception as e:
            print(f"   Warning: Volume boost failed: {e}")
            return audio

class EnhancedAudacityTruncateSilence:
    """
    Enhanced Audacity-style Truncate Silence processor with improved performance and accuracy.
    """
    
    def __init__(self):
        self.default_config = {
            # Core Audacity parameters (exact match)
            'threshold_db': -25.0,           # Audacity default: -25 dB
            'min_duration_sec': 0.5,         # Minimum silence duration to process (seconds)
            'truncate_to_sec': 0.5,          # Duration to truncate silence to (seconds)
            
            # Enhanced parameters for better speech processing
            'process_leading': True,         # Process leading silence
            'process_trailing': True,        # Process trailing silence
            'process_internal': True,        # Process internal silence
            'preserve_sentence_breaks': True, # Smart sentence break preservation
            'sentence_break_min_sec': 0.8,   # Minimum duration for sentence break
            'sentence_break_max_sec': 3.0,   # Maximum duration for sentence break
            'sentence_break_keep_ratio': 0.6, # Ratio of sentence break to keep (60%)
            'edge_protection_sec': 1.0,      # Protect silence near audio edges
            'debug_mode': False,             # Enable detailed logging
            'seek_step_ms': 10,              # Detection precision (10ms like Audacity)
            'fade_duration_ms': 5,           # Short fade to prevent clicks
            
            # Performance optimization
            'use_fast_detection': True,      # Use optimized silence detection
            'chunk_processing': True,        # Process in chunks for large files
            'max_chunk_size_sec': 300,       # Maximum chunk size (5 minutes)
        }
    
    def _fast_silence_detection(self, audio: AudioSegment, threshold_db: float, 
                               min_duration_ms: int, seek_step_ms: int = 10) -> List[Tuple[int, int]]:
        """
        Optimized silence detection using numpy for better performance.
        """
        try:
            # Convert to numpy array for faster processing
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            if audio.channels == 2:
                samples = samples.reshape((-1, 2))
                # Use RMS of both channels for stereo
                samples_rms = np.sqrt(np.mean(samples**2, axis=1))
            else:
                samples_rms = np.abs(samples)
            
            # Calculate threshold in linear scale
            max_possible = 2**(audio.sample_width * 8 - 1)
            threshold_linear = max_possible * (10**(threshold_db / 20))
            
            # Find silence regions
            frame_rate = audio.frame_rate
            step_samples = int(seek_step_ms * frame_rate / 1000)
            min_silence_samples = int(min_duration_ms * frame_rate / 1000)
            
            silence_mask = samples_rms < threshold_linear
            
            # Find contiguous silence regions
            silence_regions = []
            in_silence = False
            silence_start = 0
            
            for i in range(0, len(silence_mask), step_samples):
                chunk_end = min(i + step_samples, len(silence_mask))
                chunk_is_silent = np.all(silence_mask[i:chunk_end])
                
                if chunk_is_silent and not in_silence:
                    # Start of silence
                    silence_start = i
                    in_silence = True
                elif not chunk_is_silent and in_silence:
                    # End of silence
                    silence_duration_samples = i - silence_start
                    if silence_duration_samples >= min_silence_samples:
                        start_ms = int(silence_start * 1000 / frame_rate)
                        end_ms = int(i * 1000 / frame_rate)
                        silence_regions.append((start_ms, end_ms))
                    in_silence = False
            
            # Handle silence at end of file
            if in_silence:
                silence_duration_samples = len(silence_mask) - silence_start
                if silence_duration_samples >= min_silence_samples:
                    start_ms = int(silence_start * 1000 / frame_rate)
                    end_ms = len(audio)
                    silence_regions.append((start_ms, end_ms))
            
            return silence_regions
            
        except Exception as e:
            print(f"Warning: Fast detection failed, using fallback: {e}")
            # Fallback to pydub detection
            return detect_silence(
                audio,
                min_silence_len=min_duration_ms,
                silence_thresh=threshold_db,
                seek_step=seek_step_ms
            )
    
    def _analyze_speech_context(self, audio: AudioSegment, start_ms: int, end_ms: int,
                              context_window_ms: int = 300) -> Dict:
        """
        Enhanced context analysis for better sentence break detection.
        """
        try:
            duration = end_ms - start_ms
            
            # Get context segments
            pre_start = max(0, start_ms - context_window_ms)
            post_end = min(len(audio), end_ms + context_window_ms)
            
            before_segment = audio[pre_start:start_ms] if start_ms > pre_start else None
            after_segment = audio[end_ms:post_end] if end_ms < post_end else None
            
            # Analyze levels
            before_level = before_segment.dBFS if before_segment else float('-inf')
            after_level = after_segment.dBFS if after_segment else float('-inf')
            
            # Calculate context metrics
            level_difference = abs(before_level - after_level) if before_level != float('-inf') and after_level != float('-inf') else 0
            min_context_level = min(before_level, after_level) if before_level != float('-inf') and after_level != float('-inf') else float('-inf')
            
            # Determine if this is a sentence break
            is_sentence_break = (
                0.8 <= duration / 1000 <= 3.0 and  # Duration range for sentence breaks
                min_context_level > -50.0 and      # Both sides have reasonable speech levels
                level_difference < 8.0              # Levels are relatively consistent
            )
            
            # Determine if this is a natural pause
            is_natural_pause = (
                0.3 <= duration / 1000 <= 1.5 and  # Shorter duration for natural pauses
                min_context_level > -40.0 and      # Strong speech context
                level_difference < 5.0              # Very consistent levels
            )
            
            return {
                'duration_sec': duration / 1000,
                'before_level_db': before_level,
                'after_level_db': after_level,
                'level_difference': level_difference,
                'min_context_level': min_context_level,
                'is_sentence_break': is_sentence_break,
                'is_natural_pause': is_natural_pause,
                'context_strength': max(0, min(100, min_context_level + 60))  # 0-100 scale
            }
            
        except Exception as e:
            return {
                'duration_sec': (end_ms - start_ms) / 1000,
                'is_sentence_break': False,
                'is_natural_pause': False,
                'context_strength': 0,
                'error': str(e)
            }

    def _detect_leading_silence_numpy(self, audio: AudioSegment, threshold_db: float) -> int:
        """
        More accurate leading silence detection using numpy - FIXED VERSION
        """
        try:
            # Use numpy-based detection instead of pydub's detect_leading_silence
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            frame_rate = audio.frame_rate
            
            if audio.channels == 2:
                samples = samples.reshape((-1, 2))
                # Use RMS for stereo - more accurate than max
                level_samples = np.sqrt(np.mean(samples**2, axis=1))
            else:
                level_samples = np.abs(samples)
            
            # Calculate threshold in linear scale
            max_possible = 2**(audio.sample_width * 8 - 1)
            threshold_linear = max_possible * (10**(threshold_db / 20))
            
            # Find leading silence
            leading_silence_samples = 0
            for i in range(len(level_samples)):
                if level_samples[i] < threshold_linear:
                    leading_silence_samples += 1
                else:
                    break
            
            leading_silence_ms = int(leading_silence_samples * 1000 / frame_rate)
            
            return leading_silence_ms
            
        except Exception as e:
            print(f"   Numpy detection failed: {e}, using pydub fallback")
            return detect_leading_silence(audio, silence_threshold=threshold_db)
            
    def _detect_trailing_silence_numpy(self, audio: AudioSegment, threshold_db: float) -> int:
        """
        More accurate trailing silence detection using numpy
        """
        try:
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            frame_rate = audio.frame_rate
            
            if audio.channels == 2:
                samples = samples.reshape((-1, 2))
                level_samples = np.sqrt(np.mean(samples**2, axis=1))
            else:
                level_samples = np.abs(samples)
            
            max_possible = 2**(audio.sample_width * 8 - 1)
            threshold_linear = max_possible * (10**(threshold_db / 20))
            
            # Scan backwards from the end
            trailing_silence_samples = 0
            for i in range(len(level_samples) - 1, -1, -1):
                if level_samples[i] < threshold_linear:
                    trailing_silence_samples += 1
                else:
                    break
            
            trailing_silence_ms = int(trailing_silence_samples * 1000 / frame_rate)
            
            return trailing_silence_ms
            
        except Exception as e:
            print(f"   Numpy trailing detection failed: {e}")
            return 0
        
    def _process_leading_trailing_silence(self, audio: AudioSegment, config: Dict) -> AudioSegment:
        """
        Improved leading and trailing silence processing - NUMPY-SAFE VERSION
        """
        try:
            threshold_db = config['threshold_db']
            truncate_to_ms = int(config['truncate_to_sec'] * 1000)
            min_duration_ms = int(config['min_duration_sec'] * 1000)
            
            processed_audio = audio
            total_removed = 0
            
            process_leading = config.get('process_leading', False)
            process_trailing = config.get('process_trailing', False)
            
            print(f"   🔍 _process_leading_trailing called with:")
            print(f"      - process_leading: {process_leading}")
            print(f"      - process_trailing: {process_trailing}")
            
            if not (process_leading or process_trailing):
                print("   ⭕ Skipping leading/trailing silence (both disabled)")
                return processed_audio
                
            print("   🔄 Processing leading/trailing silence...")
            
            # แปลงเป็น numpy array เพื่อความปลอดภัย
            samples = np.array(processed_audio.get_array_of_samples(), dtype=np.int16)
            sample_rate = processed_audio.frame_rate
            sample_width = processed_audio.sample_width
            channels = processed_audio.channels
            
            if channels == 2:
                samples = samples.reshape((-1, 2))
            
            # ✅ Process leading silence
            if process_leading:
                print("   ✂️  Processing LEADING silence...")
                
                # ✅ ใช้ numpy detection ที่แม่นยำกว่า
                leading_silence_ms = self._detect_leading_silence_numpy(processed_audio, threshold_db)
                
                print(f"   🔍 DEBUG - Detected leading silence: {leading_silence_ms}ms (numpy method)")
                print(f"   🔍 DEBUG - Minimum required: {min_duration_ms}ms")
                
                if leading_silence_ms >= min_duration_ms:
                    keep_leading_ms = min(truncate_to_ms, leading_silence_ms)
                    trim_amount = leading_silence_ms - keep_leading_ms
                    
                    print(f"   🔍 DEBUG - Will keep: {keep_leading_ms}ms")
                    print(f"   🔍 DEBUG - Will trim: {trim_amount}ms")
                    
                    if trim_amount > 0:
                        # คำนวณตำแหน่ง sample
                        trim_samples = int(trim_amount * sample_rate / 1000)
                        
                        if trim_samples < len(samples):
                            # ตัดส่วน silence ออก (NOTE: No need to add back silence, just keep less of it)
                            samples = samples[trim_samples:]
                            
                            total_removed += int(trim_amount)
                            print(f"   ✅ Trimmed {int(trim_amount)}ms from leading silence by keeping only {keep_leading_ms}ms")
                else:
                    print(f"   ℹ️ Leading silence too short ({leading_silence_ms}ms), not trimming")
            else:
                print("   ⭕ SKIPPED leading silence (disabled)")
            
            # ✅ Process trailing silence
            if process_trailing:
                print("   ✂️  Processing TRAILING silence...")
                
                # สร้าง temp audio จาก samples ปัจจุบัน
                if channels == 2:
                    temp_samples = samples.flatten()
                else:
                    temp_samples = samples
                    
                temp_audio = AudioSegment(
                    temp_samples.tobytes(),
                    frame_rate=sample_rate,
                    sample_width=sample_width,
                    channels=channels
                )
                
                # ✅ ใช้ numpy detection
                trailing_silence_ms = self._detect_trailing_silence_numpy(temp_audio, threshold_db)
                
                print(f"   🔍 DEBUG - Detected trailing silence: {trailing_silence_ms}ms (numpy method)")
                print(f"   🔍 DEBUG - Minimum required: {min_duration_ms}ms")

                if trailing_silence_ms >= min_duration_ms:
                    keep_trailing_ms = min(truncate_to_ms, trailing_silence_ms)
                    trim_amount = trailing_silence_ms - keep_trailing_ms

                    print(f"   🔍 DEBUG - Will keep: {keep_trailing_ms}ms")
                    print(f"   🔍 DEBUG - Will trim: {trim_amount}ms")

                    if trim_amount > 0:
                        # คำนวณตำแหน่ง sample
                        trim_samples_count = int(trim_amount * sample_rate / 1000)
                        
                        if trim_samples_count < len(samples):
                            # ตัดส่วน silence ออก (from the end)
                            samples = samples[:-trim_samples_count]
                            
                            total_removed += int(trim_amount)
                            print(f"   ✅ Trimmed {int(trim_amount)}ms from trailing silence by keeping only {keep_trailing_ms}ms")
                else:
                    print(f"   ℹ️ Trailing silence too short ({trailing_silence_ms}ms), not trimming")
            else:
                print("   ⭕ SKIPPED trailing silence (disabled)")

            # แปลงกลับเป็น AudioSegment
            if channels == 2:
                final_samples = samples.flatten()
            else:
                final_samples = samples
                
            processed_audio = AudioSegment(
                final_samples.tobytes(),
                frame_rate=sample_rate,
                sample_width=sample_width,
                channels=channels
            )
            
            if total_removed > 0:
                print(f"   ✅ Total edge silence removed: {total_removed/1000:.2f}s")
            else:
                print(f"   ℹ️ No edge silence was removed")
            
            print(f"   🔍 Final audio length: {len(processed_audio)} ms")
            return processed_audio
            
        except Exception as e:
            import traceback
            print(f"   ⚠️ Leading/trailing processing error: {e}")
            print(f"   📋 Full traceback:")
            print(traceback.format_exc())
            print("   Falling back to original audio")
            return audio

    def _alternative_trailing_silence_detection(self, audio: AudioSegment, threshold_db: float, min_duration_ms: int) -> int:
        """
        Alternative method to detect trailing silence without reversing the entire audio.
        Uses numpy for more efficient processing.
        """
        try:
            if not ADVANCED_PROCESSING:
                return 0
                
            # Convert to numpy for analysis
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            if audio.channels == 2:
                samples = samples.reshape((-1, 2))
                # Use RMS of both channels
                samples = np.sqrt(np.mean(samples**2, axis=1))
            else:
                samples = np.abs(samples)
            
            # Calculate threshold in linear scale
            max_possible = 2**(audio.sample_width * 8 - 1)
            threshold_linear = max_possible * (10**(threshold_db / 20))
            
            # Find trailing silence by scanning backwards
            frame_rate = audio.frame_rate
            min_silence_samples = int(min_duration_ms * frame_rate / 1000)
            
            # Scan from end backwards
            silence_length = 0
            for i in range(len(samples) - 1, -1, -1):
                if samples[i] < threshold_linear:
                    silence_length += 1
                else:
                    break
            
            # Convert back to milliseconds
            silence_ms = int(silence_length * 1000 / frame_rate)
            return silence_ms if silence_ms >= min_duration_ms else 0
            
        except Exception as e:
            print(f"   Alternative trailing detection failed: {e}")
            return 0    
    def _process_internal_silence_numpy(self, audio: AudioSegment, config: Dict) -> AudioSegment:
        """
        High-performance internal silence processing using numpy arrays.
        """
        try:
            threshold_db = config['threshold_db']
            min_duration_ms = int(config['min_duration_sec'] * 1000)
            truncate_to_ms = int(config['truncate_to_sec'] * 1000)
            edge_protection_ms = int(config.get('edge_protection_sec', 1.0) * 1000)
            use_fast = config.get('use_fast_detection', True)
            
            # Detect silence segments
            if use_fast:
                silence_segments = self._fast_silence_detection(
                    audio, threshold_db, min_duration_ms, config.get('seek_step_ms', 10)
                )
            else:
                silence_segments = detect_silence(
                    audio, min_silence_len=min_duration_ms,
                    silence_thresh=threshold_db, seek_step=config.get('seek_step_ms', 10)
                )
            
            if not silence_segments:
                if config.get('debug_mode', False):
                    print("   No internal silence segments found")
                return audio
            
            if config.get('debug_mode', False):
                print(f"   Found {len(silence_segments)} silence segments")
            
            # Convert to numpy for processing
            samples = np.array(audio.get_array_of_samples(), dtype=np.int16)
            sample_rate = audio.frame_rate
            channels = audio.channels
            sample_width = audio.sample_width
            
            if channels == 2:
                samples = samples.reshape((-1, 2))
            
            samples_per_ms = sample_rate / 1000
            
            # Process segments from end to beginning to maintain positions
            total_removed_ms = 0
            segments_processed = 0
            segments_preserved = 0
            
            # Sort by start position (descending) for reverse processing
            for start_ms, end_ms in sorted(silence_segments, reverse=True):
                try:
                    silence_duration_ms = end_ms - start_ms
                    
                    # Skip segments below minimum duration
                    if silence_duration_ms < min_duration_ms:
                        continue
                    
                    # Skip segments near edges (protection)
                    if start_ms < edge_protection_ms or (len(audio) - end_ms) < edge_protection_ms:
                        segments_preserved += 1
                        if config.get('debug_mode', False):
                            edge_type = "start" if start_ms < edge_protection_ms else "end"
                            print(f"   Protected edge segment ({edge_type}): {silence_duration_ms/1000:.2f}s")
                        continue
                    
                    # Analyze context for intelligent processing
                    context = self._analyze_speech_context(audio, start_ms, end_ms)
                    
                    # Determine target duration based on context
                    if config.get('preserve_sentence_breaks', True) and context['is_sentence_break']:
                        # Handle sentence breaks with special care
                        keep_ratio = config.get('sentence_break_keep_ratio', 0.6)
                        target_duration_ms = max(
                            int(silence_duration_ms * keep_ratio),
                            truncate_to_ms
                        )
                        target_duration_ms = min(target_duration_ms, int(config.get('sentence_break_max_sec', 3.0) * 1000))
                        
                        if config.get('debug_mode', False):
                            print(f"   Sentence break: {silence_duration_ms/1000:.2f}s → {target_duration_ms/1000:.2f}s")
                    
                    elif context['is_natural_pause']:
                        # Natural pauses: reduce moderately
                        target_duration_ms = max(int(silence_duration_ms * 0.7), truncate_to_ms)
                        
                        if config.get('debug_mode', False):
                            print(f"   Natural pause: {silence_duration_ms/1000:.2f}s → {target_duration_ms/1000:.2f}s")
                    
                    else:
                        # Regular silence: truncate to target
                        target_duration_ms = truncate_to_ms
                        
                        if config.get('debug_mode', False):
                            print(f"   Regular silence: {silence_duration_ms/1000:.2f}s → {target_duration_ms/1000:.2f}s")
                    
                    # Apply processing if reduction is significant
                    if target_duration_ms < silence_duration_ms - 50:  # Only if saving >50ms
                        # Calculate sample positions
                        start_sample = int(start_ms * samples_per_ms)
                        end_sample = int(end_ms * samples_per_ms)
                        
                        # Calculate how many samples to remove
                        samples_to_remove = end_sample - start_sample - int(target_duration_ms * samples_per_ms)
                        
                        # Define slices
                        before_slice = samples[:start_sample]
                        after_slice = samples[start_sample + samples_to_remove:]
                        
                        # Remove samples by concatenating slices
                        if channels == 2:
                            samples = np.concatenate([before_slice, after_slice], axis=0)
                        else:
                            samples = np.concatenate([before_slice, after_slice])
                        
                        removed_ms = silence_duration_ms - target_duration_ms
                        total_removed_ms += removed_ms
                        segments_processed += 1
                    
                    else:
                        segments_preserved += 1
                        if config.get('debug_mode', False):
                            print(f"   Preserved: {silence_duration_ms/1000:.2f}s (minimal benefit)")
                
                except Exception as e:
                    print(f"   Warning: Error processing segment {start_ms}-{end_ms}: {e}")
                    segments_preserved += 1
                    continue
            
            # Convert back to AudioSegment
            if channels == 2:
                samples_flat = samples.flatten()
            else:
                samples_flat = samples
            
            processed_audio = AudioSegment(
                samples_flat.tobytes(),
                frame_rate=sample_rate,
                sample_width=sample_width,
                channels=channels
            )
            
            # Add subtle fade to prevent clicks if requested
            fade_ms = config.get('fade_duration_ms', 5)
            if fade_ms > 0 and len(processed_audio) > fade_ms * 2:
                processed_audio = processed_audio.fade_in(fade_ms).fade_out(fade_ms)
            
            # Summary
            if segments_processed > 0 or config.get('debug_mode', False):
                print(f"   Internal silence processed: {segments_processed} segments")
                print(f"   Segments preserved: {segments_preserved}")
                print(f"   Time saved: {total_removed_ms/1000:.2f}s")
            
            return processed_audio
            
        except Exception as e:
            print(f"Warning: Internal silence processing failed: {e}")
            return audio
    
    def _chunk_process_large_audio(self, audio: AudioSegment, config: Dict) -> AudioSegment:
        """
        Processes very large audio files in chunks to manage memory usage.
        """
        try:
            max_chunk_sec = config.get('max_chunk_size_sec', 300)  # 5 minutes
            max_chunk_ms = max_chunk_sec * 1000
            
            if len(audio) <= max_chunk_ms:
                # Small enough to process normally
                return self._process_internal_silence_numpy(audio, config)
            
            # Process in overlapping chunks
            chunk_duration = max_chunk_ms
            overlap_ms = 5000  # 5 second overlap
            
            processed_chunks = []
            pos = 0
            
            print(f"   Processing large file in chunks ({len(audio)/1000/60:.1f} minutes total)")
            
            while pos < len(audio):
                chunk_end = min(pos + chunk_duration, len(audio))
                chunk = audio[pos:chunk_end]
                
                # Process chunk
                processed_chunk = self._process_internal_silence_numpy(chunk, config)
                
                # Handle overlap
                if processed_chunks and overlap_ms > 0:
                    # Remove overlap from previous chunk
                    prev_chunk = processed_chunks[-1]
                    if len(prev_chunk) > overlap_ms:
                        prev_chunk = prev_chunk[:-overlap_ms//2]
                        processed_chunks[-1] = prev_chunk
                    
                    # Remove overlap from current chunk
                    if len(processed_chunk) > overlap_ms:
                        processed_chunk = processed_chunk[overlap_ms//2:]
                
                processed_chunks.append(processed_chunk)
                
                pos += chunk_duration - overlap_ms
                progress = min(100, (pos / len(audio)) * 100)
                print(f"\r   Chunk progress: {progress:.1f}%", end='', flush=True)
            
            print()  # New line after progress
            
            # Combine chunks
            if processed_chunks:
                result = processed_chunks[0]
                for chunk in processed_chunks[1:]:
                    result = result + chunk
                return result
            else:
                return audio
                
        except Exception as e:
            print(f"Warning: Chunk processing failed: {e}")
            return self._process_internal_silence_numpy(audio, config)
    
    def truncate_silence(self, audio: AudioSegment, config: Dict = None) -> AudioSegment:
        """
        Main truncate silence function with improved Audacity compatibility.
        """
        if config is None:
            config = self.default_config.copy()
        else:
            # ✅ FIX: ต้อง merge แบบไม่ override ค่าที่ส่งเข้ามา
            merged_config = self.default_config.copy()
            # อัพเดตเฉพาะ keys ที่มีใน config ที่ส่งเข้ามา
            for key, value in config.items():
                merged_config[key] = value
            config = merged_config
        
        try:
            start_time = time.time()
            original_duration = len(audio) / 1000.0
            
            print(f"🔇 Enhanced Audacity-style Truncate Silence:")
            print(f"   Threshold: {config['threshold_db']} dB")
            print(f"   Min Duration: {config['min_duration_sec']}s")
            print(f"   Truncate To: {config['truncate_to_sec']}s")
            print(f"   Original Duration: {original_duration:.2f}s")
            
            # ✅ เพิ่ม debug log ตรงนี้
            print(f"   🔍 DEBUG - process_leading: {config.get('process_leading', 'NOT SET')}")
            print(f"   🔍 DEBUG - process_trailing: {config.get('process_trailing', 'NOT SET')}")
            print(f"   🔍 DEBUG - process_internal: {config.get('process_internal', 'NOT SET')}")
                   
            processed_audio = audio
            
            # Step 1: Process leading and trailing silence
            should_process_edges = config.get('process_leading', True) or config.get('process_trailing', True)

            
            if should_process_edges:
                print("   🔄 Processing leading/trailing silence...")
                processed_audio = self._process_leading_trailing_silence(processed_audio, config)
            else:
                print("   ⏭️  Skipping leading/trailing silence (both disabled)")
            
            # Step 2: Process internal silence
            if config.get('process_internal', True):
                print("   🔄 Processing internal silence...")
                
                # Use chunk processing for very large files
                if config.get('chunk_processing', True) and len(processed_audio) > 600000:  # >10 minutes
                    processed_audio = self._chunk_process_large_audio(processed_audio, config)
                else:
                    processed_audio = self._process_internal_silence_numpy(processed_audio, config)
            
            # Final summary
            final_duration = len(processed_audio) / 1000.0
            time_saved = original_duration - final_duration
            processing_time = time.time() - start_time
            
            if time_saved > 0.1:
                print(f"   ✅ Duration: {original_duration:.2f}s → {final_duration:.2f}s")
                print(f"   ⏱️ Time saved: {time_saved:.2f}s ({(time_saved/original_duration)*100:.1f}%)")
                print(f"   ⚡ Processing time: {processing_time:.2f}s")
            else:
                print(f"   ℹ️ Duration maintained: {final_duration:.2f}s (minimal changes)")
            
            return processed_audio
            
        except Exception as e:
            print(f"Error in enhanced truncate_silence: {e}")
            return audio

def apply_enhanced_audacity_silence_processing(audio: AudioSegment, silence_config: Dict) -> AudioSegment:
    """
    Apply enhanced Audacity-style silence processing - CORRECTED VERSION
    """
    if not silence_config or not silence_config.get('enabled', True):
        print("   Silence processing: Disabled")
        return audio
    
    try:
        processor = EnhancedAudacityTruncateSilence()
        
        # Convert config format
        enhanced_config = {
            'threshold_db': silence_config.get('threshold_db', -25.0),
            'min_duration_sec': silence_config.get('min_silence_to_process', 500) / 1000.0,
            'truncate_to_sec': silence_config.get('max_internal_silence_ms', 500) / 1000.0,
            
            # ✅ ใช้ค่า process_leading/trailing โดยตรงเพื่อความแม่นยำ
            'process_leading': silence_config.get('process_leading', True),
            'process_trailing': silence_config.get('process_trailing', True),
            
            'preserve_sentence_breaks': silence_config.get('sentence_break_detection', True),
            'sentence_break_keep_ratio': silence_config.get('sentence_break_keep_ratio', 0.6),
            'edge_protection_sec': max(0.5, silence_config.get('padding', 150) / 1000.0 * 5),
            'debug_mode': silence_config.get('debug_mode', False),
            'use_fast_detection': True,
            'chunk_processing': True,
            'fade_duration_ms': 3
        }
        
        return processor.truncate_silence(audio, enhanced_config)
        
    except Exception as e:
        print(f"   ⚠️ Error: Enhanced silence processing failed: {e}")
        return audio

class SystemMonitor:
    """System performance monitoring"""
    
    def __init__(self):
        self.start_time = time.time()
        self.peak_memory = 0
        self.cpu_usage = []
        
    def update(self):
        if SYSTEM_MONITORING:
            current_memory = psutil.virtual_memory().percent
            current_cpu = psutil.cpu_percent()
            self.peak_memory = max(self.peak_memory, current_memory)
            self.cpu_usage.append(current_cpu)
    
    def get_stats(self):
        elapsed = time.time() - self.start_time
        avg_cpu = sum(self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0
        return {
            'elapsed_time': elapsed,
            'peak_memory_percent': self.peak_memory,
            'average_cpu_percent': avg_cpu,
            'threads_used': THREAD_POOL_SIZE
        }

class FastProgressBar:
    """Fast and efficient progress bar"""
    
    def __init__(self, total: int, prefix: str = "Processing", width: int = 40):
        self.total = total
        self.current = 0
        self.prefix = prefix
        self.width = width
        self.start_time = time.time()
        self.last_update = 0
        
    def update(self, current: int, force: bool = False):
        now = time.time()
        if not force and (now - self.last_update) < 0.1:
            return
            
        self.current = current
        self.last_update = now
        
        percent = (current / self.total) * 100 if self.total > 0 else 0
        filled = int(self.width * current // self.total) if self.total > 0 else 0
        bar = '█' * filled + '▒' * (self.width - filled)
        
        elapsed = now - self.start_time
        if current > 0 and elapsed > 0:
            rate = current / elapsed
            eta = (self.total - current) / rate if rate > 0 else 0
            eta_str = f"ETA: {eta:.0f}s" if eta < 3600 else f"ETA: {eta/3600:.1f}h"
        else:
            eta_str = "ETA: --"
            
        print(f'\r{Colors.CYAN}{self.prefix}{Colors.RESET}: |{Colors.GREEN}{bar}{Colors.RESET}| '
              f'{percent:.0f}% ({current}/{self.total}) {eta_str}', end='', flush=True)
        
        if current == self.total:
            print(f' {Colors.GREEN}✓ Complete!{Colors.RESET}')

class AdvancedEQ:
    """Advanced parametric EQ system"""
    
    @staticmethod
    def apply_parametric_eq(audio: AudioSegment, eq_bands: Dict) -> AudioSegment:
        """Apply professional parametric EQ"""
        if not ADVANCED_PROCESSING or not eq_bands:
            return audio
        
        try:
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            sample_rate = audio.frame_rate
            
            if audio.channels == 2:
                samples = samples.reshape((-1, 2))
                
            processed_samples = samples.copy()
            
            for band_name, params in eq_bands.items():
                freq = params['freq']
                gain = params['gain']
                q = params['q']
                
                if abs(gain) < 0.1:
                    continue
                    
                w0 = 2 * np.pi * freq / sample_rate
                alpha = np.sin(w0) / (2 * q)
                A = 10**(gain / 40)
                
                # Peaking EQ coefficients
                b0 = 1 + alpha * A
                b1 = -2 * np.cos(w0)
                b2 = 1 - alpha * A
                a0 = 1 + alpha / A
                a1 = -2 * np.cos(w0)
                a2 = 1 - alpha / A
                
                b = [b0/a0, b1/a0, b2/a0]
                a = [1, a1/a0, a2/a0]
                
                if audio.channels == 2:
                    for ch in range(2):
                        processed_samples[:, ch] = signal.lfilter(b, a, processed_samples[:, ch])
                else:
                    processed_samples = signal.lfilter(b, a, processed_samples)
            
            if audio.channels == 2:
                processed_samples = processed_samples.flatten()
                
            processed_samples = np.clip(processed_samples, -32768, 32767)
            processed_samples = processed_samples.astype(np.int16)
            
            return AudioSegment(
                processed_samples.tobytes(),
                frame_rate=sample_rate,
                sample_width=audio.sample_width,
                channels=audio.channels
            )
            
        except Exception as e:
            print(f"Warning: EQ processing failed: {e}")
            return audio

class ParallelCompressor:
    """Handles parallel compression of audio chunks using multiprocessing"""
    
    def __init__(self, threshold: float = -20.0, ratio: float = 2.0, 
                 chunk_size_ms: int = 1000, max_workers: int = None):
        self.threshold = threshold
        self.ratio = ratio
        self.chunk_size_ms = chunk_size_ms
        self.max_workers = max_workers or min(multiprocessing.cpu_count(), 8)
    
    @staticmethod
    def _compress_chunk(chunk_data: Tuple[AudioSegment, Dict]) -> AudioSegment:
        """Compress a single audio chunk"""
        chunk, settings = chunk_data
        if not chunk:
            return chunk
        return compress_dynamic_range(
            chunk,
            threshold=settings['threshold'],
            ratio=settings['ratio']
        )
    
    def process(self, audio: AudioSegment, compression_settings: Dict, debug_speed: bool = False) -> AudioSegment:
        """Process audio in parallel chunks - FIXED: maintains chunk order"""
        start_time = time.time()
        
        if len(audio) < self.chunk_size_ms:
            print("📄 Processing short audio segment...")
            return compress_dynamic_range(
                audio,
                threshold=compression_settings['threshold'],
                ratio=compression_settings['ratio']
            )
        
        # Split audio into chunks for parallel processing
        print(f"📊 Splitting audio into chunks of {self.chunk_size_ms}ms...")
        chunks = make_chunks(audio, self.chunk_size_ms)
        total_chunks = len(chunks)
        chunk_pairs = [(chunk, compression_settings) for chunk in chunks]
        split_time = time.time()
        
        print(f"⚡ Starting parallel processing ({total_chunks} chunks)...")
        
        # IMPORTANT: Use a dictionary to store chunk order
        chunk_results = {}
        completed = 0
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit chunks with an index to maintain order
            futures_to_index = {}
            for i, chunk_pair in enumerate(chunk_pairs):
                future = executor.submit(self._compress_chunk, chunk_pair)
                futures_to_index[future] = i
            
            progress_update = time.time()
            
            for future in as_completed(futures_to_index):
                chunk_index = futures_to_index[future]
                chunk_results[chunk_index] = future.result()
                completed += 1
                
                # Update progress bar
                now = time.time()
                if now - progress_update >= 0.1:
                    width = 40
                    filled = int(width * completed // total_chunks)
                    progress_bar = '█' * filled + '▒' * (width - filled)
                    percent = (completed / total_chunks) * 100
                    
                    elapsed = now - start_time
                    chunks_per_sec = completed / elapsed if elapsed > 0 else 0
                    eta = (total_chunks - completed) / chunks_per_sec if chunks_per_sec > 0 else 0
                    
                    print(f"\r📄 [{progress_bar}] {percent:5.1f}% | {completed}/{total_chunks} chunks | "
                          f"ETA: {eta:.1f}s", end='', flush=True)
                    progress_update = now
                
        print("\n✅ Parallel processing completed")
        
        # IMPORTANT: Combine chunks in the correct order (0, 1, 2, 3, ...)
        print("📄 Combining chunks in time order...")
        ordered_chunks = []
        for i in range(total_chunks):
            if i in chunk_results:
                ordered_chunks.append(chunk_results[i])
            else:
                print(f"⚠️ Warning: Missing chunk {i}, using original chunk")
                ordered_chunks.append(chunks[i])
        
        compress_time = time.time()

        if debug_speed:
            end_time = time.time()
            print(f"⏱️ Timings:")
            print(f"   - Chunk splitting: {split_time - start_time:.2f}s")
            print(f"   - Parallel processing: {compress_time - split_time:.2f}s")
            print(f"   - Chunk combination: {end_time - compress_time:.2f}s")
            print(f"   - Total: {end_time - start_time:.2f}s")
            
        # Combine chunks back into a single AudioSegment
        print("🎵 Creating final AudioSegment...")
        if ordered_chunks:
            result = ordered_chunks[0]
            for chunk in ordered_chunks[1:]:
                result = result + chunk
            return result
        else:
            print("❌ Error: No processed chunks available")
            return audio

class UnifiedAudioProcessor:
    """Unified audio processor with enhanced peak control and improved processing order"""
    
    def __init__(self, speed_config: Dict = None):
        self.monitor = SystemMonitor()
        self.logger = self._setup_logger()
        self.speed_config = speed_config or SPEED_PRESETS['normal']
        self.peak_detector = PeakDetector()
        self.hard_limiter = HardLimiter()

    def analyze_stereo_channels(self, audio: AudioSegment) -> Dict:
        """Analyzes audio channels and provides recommendations for stereo conversion."""
        result = {
            'left_db': float('-inf'),
            'right_db': float('-inf'),
            'left_rms': 0,
            'right_rms': 0,
            'left_silent': True,
            'right_silent': True,
            'active_channels': 'unknown',
            'recommendation': 'keep_as_is'
        }

        if audio.channels != 2:
            result['active_channels'] = f'mono ({audio.channels} channel)'
            result['recommendation'] = 'convert_to_stereo'
            return result

        # Split left-right channels
        channels = audio.split_to_mono()
        left, right = channels[0], channels[1]

        # Analyze each channel
        result['left_db'] = left.dBFS
        result['right_db'] = right.dBFS
        result['left_rms'] = left.rms
        result['right_rms'] = right.rms

        # Check for silence (using a -50 dBFS threshold)
        silence_threshold = -50
        result['left_silent'] = left.dBFS < silence_threshold
        result['right_silent'] = right.dBFS < silence_threshold

        # Analyze and provide recommendations
        if result['left_silent'] and result['right_silent']:
            result['active_channels'] = 'both channels silent'
            result['recommendation'] = 'keep_as_is'
        elif result['left_silent']:
            result['active_channels'] = 'right channel only'
            result['recommendation'] = 'use_right_as_source'
        elif result['right_silent']:
            result['active_channels'] = 'left channel only'
            result['recommendation'] = 'use_left_as_source'
        else:
            # Compare volume levels between the two channels
            db_difference = abs(result['left_db'] - result['right_db'])
            if db_difference > 20:  # Difference is more than 20dB
                if result['left_db'] > result['right_db']:
                    result['active_channels'] = 'left channel dominant'
                    result['recommendation'] = 'use_left_as_source'
                else:
                    result['active_channels'] = 'right channel dominant'
                    result['recommendation'] = 'use_right_as_source'
            else:
                result['active_channels'] = 'both channels active'
                result['recommendation'] = 'keep_as_is'

        return result
        
    def _setup_logger(self) -> logging.Logger:
        """Setup optimized logging"""
        logger = logging.getLogger('UnifiedAudioProcessor')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        return logger

    async def process_audio_async(self, input_file: str, output_file: str, settings: Dict, args: argparse.Namespace) -> bool:
        """Enhanced async audio processing pipeline with improved peak handling and processing order"""
        try:
            print(f"\n🚀 Processing: {os.path.basename(input_file)}")
            print("📁 Loading audio file...")
            start_time = time.time()
            
            # Load audio file
            audio = AudioSegment.from_file(input_file)
            load_time = time.time() - start_time
            
            print(f"📊 Duration: {len(audio)/1000:.1f}s | Channels: {audio.channels}")
            print(f"⏱️ Load time: {load_time:.2f}s | Chunk Size: {CHUNK_SIZE_MS}ms | Threads: {THREAD_POOL_SIZE}")
            
            print(f"\n🎵 Enhanced Audio Processing Pipeline v5.8")
            print("==================================================")
            
            # 1. Channel Processing & Conversion (priority)
            print("1️⃣ Channel Processing & Conversion:")
            if audio.channels == 2:
                analysis = self.analyze_stereo_channels(audio)
                print(f"   Left: {analysis['left_db']:.1f} dBFS | Right: {analysis['right_db']:.1f} dBFS")
                
                # Display additional info for debugging
                print(f"   Recommendation: {analysis['recommendation']}")
                print(f"   Active Channels: {analysis['active_channels']}")
                
                if analysis['recommendation'] == 'use_left_as_source' or args.use_left_channel:
                    print("   🔀 Auto-merged: Using left channel as source for both channels")
                    left_channel = audio.split_to_mono()[0]
                    # Create stereo by copying the left channel to the right channel (L=R)
                    audio = AudioSegment.from_mono_audiosegments(left_channel, left_channel)
                    print("   Result: L=L, R=L (Right channel gets audio from the left channel)")
                elif analysis['recommendation'] == 'use_right_as_source' or args.use_right_channel:
                    print("   🔀 Auto-merged: Using right channel as source for both channels")
                    right_channel = audio.split_to_mono()[1]
                    # Create stereo by copying the right channel to the left channel (L=R)
                    audio = AudioSegment.from_mono_audiosegments(right_channel, right_channel)
                    print("   Result: L=R, R=R (Left channel gets audio from the right channel)")
                else:
                    print("   ✅ Both channels have signal - keeping as stereo")
                    print("   Result: L=L, R=R (original channels preserved)")
            elif audio.channels == 1:
                print("   🔀 Converting mono to stereo...")
                # Convert mono to stereo (L=R=mono)
                audio = audio.set_channels(2)
                print("   Result: L=mono, R=mono (both channels get the same audio)")
            
            # 2. Peak Detection & Analysis (NEW - before normalization)
            print("2️⃣ Peak Detection & Analysis:")
            peak_analysis = self.peak_detector.analyze_peaks(audio, threshold_db=-1.0)
            
            print(f"   Peak Level: {peak_analysis['peak_dbfs']:.1f} dBFS")
            print(f"   RMS Level: {peak_analysis['rms_dbfs']:.1f} dBFS")
            print(f"   Dynamic Range: {peak_analysis['crest_factor']:.1f} dB")
            
            if peak_analysis['peak_percentage'] > 0:
                print(f"   High Peaks: {peak_analysis['peak_percentage']:.2f}% of samples")
            
            # 3. Pre-Normalization Hard Limiting (NEW - very important!)
            print("3️⃣ Pre-Normalization Peak Control:")
            if peak_analysis['needs_limiting']:
                print(f"   🚨 High peaks detected - applying hard limiter")
                print(f"   Amount to limit: {peak_analysis['limiting_amount_db']:.1f} dB")
                
                limiter_threshold = peak_analysis['recommended_limiter_threshold']
                audio = self.hard_limiter.apply_hard_limit(
                    audio, 
                    threshold_db=limiter_threshold,
                    release_ms=5.0
                )
                
                # Re-analyze after limiting
                post_limit_analysis = self.peak_detector.analyze_peaks(audio)
                print(f"   ✅ Peak after limiting: {post_limit_analysis['peak_dbfs']:.1f} dBFS")
            else:
                print("   ✅ No pre-limiting needed - peaks are within acceptable range")
            
            # 4. Normalization (now safe and consistent)
            print("4️⃣ Normalization:")
            if settings.get('normalize', True):
                print("   📊 Applying normalization (peaks are now controlled)")
                pre_norm_db = audio.dBFS
                audio = normalize(audio)
                post_norm_db = audio.dBFS
                print(f"   Level change: {pre_norm_db:.1f} dB → {post_norm_db:.1f} dBFS")
            else:
                print("   Normalization: Disabled")
            
            # 5. Intelligent Volume Boost (NEW - replaces Gain Adjustment)
            print("5️⃣ Intelligent Volume Boost:")
            if 'volume_boost' in settings and settings['volume_boost']:
                audio = VolumeBoostProcessor.apply_intelligent_volume_boost(audio, settings['volume_boost'])
            elif settings.get('gain_db', 0) != 0:
                # Fallback to traditional gain adjustment
                gain_db = settings['gain_db']
                print(f"   🎚️ Using traditional gain: {gain_db:+.1f} dB")
                audio = audio + gain_db
                
                # Check if new peaks were created after gain adjustment
                post_gain_analysis = self.peak_detector.analyze_peaks(audio)
                if post_gain_analysis['needs_limiting']:
                    print(f"   ⚠️ Gain adjustment created new peaks - applying safety limiter")
                    audio = self.hard_limiter.apply_hard_limit(audio, threshold_db=-0.1)
                    final_analysis = self.peak_detector.analyze_peaks(audio)
                    print(f"   ✅ Peak after safety limiting: {final_analysis['peak_dbfs']:.1f} dBFS")
            else:
                print("   No additional volume adjustment")
            
            # 6. Compression
            print("6️⃣ Dynamic Range Compression:")
            if 'compression' in settings and settings['compression']:
                threshold = settings['compression']['threshold']
                print(f"🗜️ Applying compression (threshold: {threshold} dB)")
                compressor = ParallelCompressor(
                    threshold=threshold,
                    ratio=settings['compression']['ratio'],
                    chunk_size_ms=CHUNK_SIZE_MS,
                    max_workers=THREAD_POOL_SIZE
                )
                audio = compressor.process(audio, settings['compression'], args.debug_speed)
            
            # 7. Noise Reduction
            print("7️⃣ Noise Reduction:")
            if 'noise_reduction' in settings and settings['noise_reduction'] and ADVANCED_PROCESSING:
                print("🔇 Applying noise reduction")
                try:
                    # Convert to numpy array for noise reduction
                    samples = np.array(audio.get_array_of_samples())
                    rate = audio.frame_rate
                    
                    # Process in chunks to save memory
                    chunk_duration = 30  # seconds
                    chunk_size = int(rate * chunk_duration)
                    total_samples = len(samples)
                    processed_samples = np.zeros_like(samples)
                    
                    print("   Processing in chunks for memory efficiency...")
                    
                    for i in range(0, total_samples, chunk_size):
                        chunk_end = min(i + chunk_size, total_samples)
                        chunk = samples[i:chunk_end]
                        
                        # Apply noise reduction to chunk
                        reduced_chunk = nr.reduce_noise(
                            y=chunk.astype(float),
                            sr=rate,
                            prop_decrease=settings['noise_reduction']['strength'],
                            n_jobs=1
                        )
                        
                        processed_samples[i:chunk_end] = reduced_chunk.astype(np.int16)
                        
                        # Show progress
                        progress = (chunk_end / total_samples) * 100
                        print(f"\r   Progress: {progress:.1f}%", end='', flush=True)
                    
                    print("\n   ✅ Noise reduction complete")
                    
                    # Convert back to AudioSegment
                    audio = audio._spawn(processed_samples)
                except Exception as e:
                    print(f"\n   ⚠️ Warning: Noise reduction failed: {str(e)}")
                    print("   Continuing without noise reduction...")
            
            # 8. Silence Processing (Enhanced with Audacity-style Truncate Silence)
            print("8️⃣ Silence Processing:")
            if 'silence_processing' in settings and settings['silence_processing']:
                audio = apply_enhanced_audacity_silence_processing(audio, settings['silence_processing'])
            
            # 9. EQ (if enabled)
            print("9️⃣ Parametric EQ:")
            if settings.get('eq_enabled', False) and 'eq_bands' in settings:
                print("🎛️ Applying EQ")
                audio = AdvancedEQ.apply_parametric_eq(audio, settings['eq_bands'])
            else:
                print("   EQ: Disabled")
            
            # 10. Final Safety Limiter (NEW)
            print("🔟 Final Safety Limiter:")
            if 'limiter' in settings and settings['limiter']:
                threshold = settings['limiter']['threshold']
                print(f"🛡️ Applying final safety limiter (threshold: {threshold} dB)")
                audio = self.hard_limiter.apply_hard_limit(audio, threshold_db=threshold)
            
            # 11. Pre-Export Volume Boost (NEW - boost volume before export)
            print("1️⃣1️⃣ Pre-Export Volume Boost:")
            pre_export_boost = settings.get('pre_export_boost', {})
            if pre_export_boost.get('enabled', True):  # Enabled by default
                boost_db = pre_export_boost.get('boost_db', 6.0)  # +6 dB default boost
                max_safe_peak = pre_export_boost.get('max_safe_peak', -0.1)
                
                print(f"📊 Boosting volume before export: {boost_db:+.1f} dB")
                
                # Analyze current level
                current_peak = self.peak_detector.analyze_peaks(audio)['peak_dbfs']
                print(f"   Current peak level: {current_peak:.1f} dBFS")
                
                # Calculate safe gain
                safe_boost = min(boost_db, max_safe_peak - current_peak - 0.5)  # Leave 0.5 dB headroom
                
                if safe_boost > 0.1:
                    print(f"   Applying safe boost: {safe_boost:+.1f} dB")
                    audio = audio + safe_boost
                    
                    # Check and apply limiter if necessary
                    final_peak = self.peak_detector.analyze_peaks(audio)['peak_dbfs']
                    if final_peak > max_safe_peak:
                        print(f"   ⚠️ Peak exceeds limit ({final_peak:.1f} dB) - applying emergency limiter")
                        audio = self.hard_limiter.apply_hard_limit(audio, threshold_db=max_safe_peak)
                        emergency_peak = self.peak_detector.analyze_peaks(audio)['peak_dbfs']
                        print(f"   ✅ Peak after emergency limiting: {emergency_peak:.1f} dBFS")
                    else:
                        print(f"   ✅ Peak after boost: {final_peak:.1f} dBFS")
                else:
                    print(f"   ⚠️ Cannot safely apply boost (headroom remaining: {max_safe_peak - current_peak:.1f} dB)")
            else:
                print("   Pre-export boost: Disabled")
            
            # Export - FIXED: Force bitrate after sample rate conversion
            output_format = settings.get('output_format', args.format if hasattr(args, 'format') and args.format else 'mp3')
            output_bitrate = settings.get('bitrate', args.bitrate if hasattr(args, 'bitrate') and args.bitrate else '192k')
            output_sample_rate = settings.get('sample_rate', args.sample_rate if hasattr(args, 'sample_rate') and args.sample_rate else None)

            print(f"\n💾 Exporting to {output_format.upper()}...")

            # Build format arguments dynamically
            format_args = {
                'format': output_format,
                'parameters': ["-q:a", "0"]
            }

            # Handle sample rate conversion FIRST
            if output_sample_rate and output_sample_rate != audio.frame_rate:
                print(f"   Converting sample rate: {audio.frame_rate}Hz → {output_sample_rate}Hz")
                audio = audio.set_frame_rate(output_sample_rate)

            # FIXED: Force bitrate AFTER sample rate conversion
            if output_format in ['mp3', 'ogg', 'm4a', 'aac'] and output_bitrate and output_bitrate.lower() != 'auto':
                format_args['bitrate'] = output_bitrate
                # Add explicit ffmpeg parameters to force exact bitrate
                format_args['parameters'].extend(["-b:a", output_bitrate])
                print(f"   Forcing bitrate: {output_bitrate}")
                
            # ALTERNATIVE: Use CBR (Constant Bit Rate) encoding - เพิ่มตรงนี้
            if output_format == 'mp3' and output_bitrate and output_bitrate.lower() != 'auto':
                format_args['parameters'].extend([
                    "-codec:a", "libmp3lame",
                    "-b:a", output_bitrate,
                    "-cbr", "1"  # Force constant bit rate
                ])
                print(f"   Forcing CBR encoding at {output_bitrate}")
    
            export_start = time.time()
            audio.export(output_file, **format_args)
            export_time = time.time() - export_start
            
            # Final stats
            end_time = time.time()
            total_time = end_time - start_time
            realtime_factor = len(audio) / (total_time * 1000)
            
            print(f"\n✅ Processing complete!")
            print(f"📁 Output: {os.path.basename(output_file)}")
            print(f"⏱️ Processing Time: {total_time:.2f}s (Export: {export_time:.2f}s)")
            print(f"⚡ Speed: {realtime_factor:.1f}x realtime")
            print(f"🎯 Final Peak Level: {audio.dBFS:.1f} dBFS")
            
            input_size = os.path.getsize(input_file) / (1024 * 1024)
            output_size = os.path.getsize(output_file) / (1024 * 1024)
            print(f"📦 File Size: {input_size:.1f}MB → {output_size:.1f}MB")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Error processing {input_file}: {str(e)}")
            if args.debug_speed:
                import traceback
                traceback.print_exc()
            return False

# Enhanced presets with improved settings
def get_enhanced_sermon_ultra_preset():
    """Enhanced sermon preset with improved Audacity-style truncate silence"""
    return {
        'name': 'Sermon Ultra with Enhanced Truncate Silence v5.8',
        'description': 'Premium sermon processing with enhanced Audacity-style silence truncation and peak control',
        'normalize': True,
        'gain_db': 8.0,
        'eq_enabled': True,
        'eq_bands': {
            'bass': {'freq': 120, 'gain': 3.0, 'q': 1.0},
            'warmth': {'freq': 400, 'gain': 5.0, 'q': 1.3},
            'clarity': {'freq': 1500, 'gain': 4.0, 'q': 1.4},
            'presence': {'freq': 2800, 'gain': 3.0, 'q': 1.2}
        },
        'compression': {
            'threshold': -30,
            'ratio': 3.0,
            'attack': 5,
            'release': 250,
            'parallel': {
                'enabled': True,
                'chunk_size': 12000,
                'threads': 6,
                'overlap_ms': 25,
                'prealloc': True,
                'cache_enabled': True
            }
        },
        'noise_reduction': {
            'strength': 0.8,
            'preserve_speech': True,
            'sensitivity': 0.5
        },
        'limiter': {
            'threshold': -1.0,
            'release': 150
        },
        'stereo_enhancement': False,
        'stereo_width': 1.4,
        'peak_control': {
            'enabled': True,
            'threshold_db': -0.5,
            'pre_normalization': True,
            'safety_limiter': True
        },
        'volume_boost': {
            'enabled': False,
            'target_lufs': -16.0,
            'max_peak_db': -0.3,
            'adaptive_gain': True,
            'preserve_dynamics': True
        },
        'silence_processing': {
            # Enhanced Audacity-style settings
            'trim_silence': True,
            'audacity_style': True,
            'threshold_db': -35.0,
            'min_silence_to_process': 750,    # 0.85 seconds
            'max_internal_silence_ms': 600,   # 0.7 seconds
            'compress_internal': True,
            'sentence_break_detection': True,
            'sentence_break_keep_ratio': 0.8, # Keep 60% of sentence breaks
            'padding': 50,
            'debug_mode': False,
            'process_leading': True,  # ✅ เปิดการตัดเสียงเงียบช่วงเริ่มต้น
            'process_trailing': True,  # เปิดการตัดเสียงเงียบช่วงท้าย
            
            # Enhanced parameters
            'use_fast_detection': True,
            'chunk_processing': True,
            'fade_duration_ms': 50,
            'edge_protection_sec': 0.8
        }
    }

# Professional quality presets with peak control and Enhanced Audacity-style truncate silence
QUALITY_PRESETS = {
    'broadcast_pro': {
        'name': 'Broadcast Professional v5.8',
        'description': 'Professional broadcast quality processing with enhanced truncate silence',
        'normalize': True,
        'gain_db': 2.0,
        'eq_enabled': False,
        'eq_bands': {
            'bass': {'freq': 200, 'gain': 1.5, 'q': 0.8},
            'mid': {'freq': 1500, 'gain': 1.0, 'q': 1.2},
            'presence': {'freq': 8000, 'gain': 2.0, 'q': 0.8}
        },
        'compression': {'threshold': -24, 'ratio': 2.5, 'attack': 5, 'release': 150},
        'noise_reduction': {'strength': 0.6, 'preserve_speech': True},
        'limiter': {'threshold': -1.0, 'release': 50},
        'stereo_enhancement': True,
        'stereo_width': 1.2,
        'peak_control': {'enabled': True, 'threshold_db': -1.0},
        'silence_processing': {
            'trim_silence': True,        
            'audacity_style': True,
            'threshold_db': -30.0,
            'min_silence_to_process': 800,
            'max_internal_silence_ms': 600,
            'process_leading': True,  # ✅ เปิดการตัดเสียงเงียบช่วงเริ่มต้น
            'process_trailing': True,  # เปิดการตัดเสียงเงียบช่วงท้าย            
            'compress_internal': True,
            'sentence_break_detection': True,
            'sentence_break_keep_ratio': 0.8,
            'debug_mode': False,
            'use_fast_detection': True,
            'chunk_processing': True
        }
    },
    
    'sermon_ultra': get_enhanced_sermon_ultra_preset(),
    
    'podcast_pro': {
        'name': 'Podcast Professional v5.8',
        'description': 'High-quality podcast processing with enhanced silence handling',
        'normalize': True,
        'gain_db': 4.0,
        'eq_enabled': False,
        'eq_bands': {
            'warmth': {'freq': 200, 'gain': 1.0, 'q': 0.9},
            'body': {'freq': 500, 'gain': 1.5, 'q': 1.0},
            'clarity': {'freq': 1500, 'gain': 3.0, 'q': 1.2},
            'presence': {'freq': 4000, 'gain': 2.0, 'q': 1.0}
        },
        'compression': {'threshold': -16, 'ratio': 4.0, 'attack': 8, 'release': 80},
        'noise_reduction': {'strength': 0.8, 'preserve_speech': True},
        'limiter': {'threshold': -0.5, 'release': 30},
        'stereo_enhancement': True,
        'stereo_width': 1.3,
        'peak_control': {'enabled': True, 'threshold_db': -1.0},
        'silence_processing': {
            'trim_silence': True,
            'audacity_style': True,
            'threshold_db': -35.0,
            'min_silence_to_process': 600,
            'max_internal_silence_ms': 400,
            'process_leading': True,  # ✅ เปิดการตัดเสียงเงียบช่วงเริ่มต้น
            'process_trailing': True,  # เปิดการตัดเสียงเงียบช่วงท้าย            
            'compress_internal': True,
            'sentence_break_detection': True,
            'sentence_break_keep_ratio': 0.7,
            'debug_mode': False,
            'use_fast_detection': True,
            'chunk_processing': True
        }
    },

    'music_mastering': {
        'name': 'Music Mastering v5.8',
        'description': 'Professional music mastering with minimal silence processing',
        'normalize': False,
        'gain_db': 0.0,
        'eq_enabled': False,
        'eq_bands': {
            'bass': {'freq': 80, 'gain': 1.0, 'q': 1.0},
            'mid': {'freq': 1500, 'gain': 0.5, 'q': 1.1},
            'presence': {'freq': 4000, 'gain': 1.0, 'q': 0.9},
            'air': {'freq': 12000, 'gain': 1.5, 'q': 0.7}
        },
        'compression': {'threshold': -12, 'ratio': 2.0, 'attack': 20, 'release': 200},
        'noise_reduction': {'strength': 0.2, 'preserve_music': True},
        'limiter': {'threshold': -0.3, 'release': 100},
        'stereo_enhancement': True,
        'stereo_width': 1.1,
        'peak_control': {'enabled': True, 'threshold_db': -0.3},
        'silence_processing': {
            'trim_silence': False,
            'audacity_style': False,
            'threshold_db': -60.0,
            'process_leading': False,  # ✅ เปิดการตัดเสียงเงียบช่วงเริ่มต้น
            'process_trailing': False,  # เปิดการตัดเสียงเงียบช่วงท้าย            
            'compress_internal': False
        }
    }
}

# Batch processing and utility functions
async def batch_process_files(patterns: List[str], settings: Dict, args: argparse.Namespace) -> None:
    all_files = []
    for pattern in patterns:
        files = glob.glob(pattern, recursive=True)
        all_files.extend(files)
    
    # Filter for supported audio files
    audio_files = [f for f in all_files 
                   if Path(f).suffix.lower() in SUPPORTED_FORMATS]
    
    if not audio_files:
        print("No audio files found matching the pattern")
        return
    
    print(f"Found {len(audio_files)} audio files to process")
    
    # Process files
    processor = UnifiedAudioProcessor()
    successful = 0
    failed = 0
    
    for input_file in audio_files:
        input_path = Path(input_file)
        
        # Generate output filename
        if args.format:
            output_ext = f".{args.format}"
        else:
            output_ext = input_path.suffix
        
        output_file = str(input_path.parent / f"{input_path.stem}_processed{output_ext}")
        
        # Skip if output exists and not overwrite
        if os.path.exists(output_file) and not args.overwrite:
            print(f"Skipping (file exists): {os.path.basename(output_file)}")
            continue
        
        # Process file
        success = await processor.process_audio_async(input_file, output_file, settings, args)
        
        if success:
            successful += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n{Colors.BOLD}Batch Processing Summary{Colors.RESET}")
    print(f"Successful: {Colors.GREEN}{successful}{Colors.RESET}")
    if failed > 0:
        print(f"Failed: {Colors.RED}{failed}{Colors.RESET}")

def analyze_channels_only(file_path: str) -> None:
    """Analyzes channels without processing - for inspection."""
    try:
        print(f"Analyzing channels in: {file_path}")
        audio = AudioSegment.from_file(file_path)
        
        print(f"\nFile: {os.path.basename(file_path)}")
        print(f"Channels: {audio.channels}")
        print(f"Duration: {len(audio)/1000:.1f} seconds")
        print(f"Sample Rate: {audio.frame_rate} Hz")
        
        if audio.channels == 2:
            processor = UnifiedAudioProcessor()
            analysis = processor.analyze_stereo_channels(audio)
            
            print(f"\n{Colors.BOLD}Channel Analysis:{Colors.RESET}")
            print("=" * 40)
            print(f"Left Channel (L):  RMS={analysis['left_rms']:.6f}, dBFS={analysis['left_db']:.1f}, Silent={analysis['left_silent']}")
            print(f"Right Channel (R): RMS={analysis['right_rms']:.6f}, dBFS={analysis['right_db']:.1f}, Silent={analysis['right_silent']}")
            
            # Visual comparison
            if analysis['left_db'] > analysis['right_db']:
                left_bars = '█' * min(20, int((analysis['left_db'] + 80) / 4))
                right_bars = '▓' * min(20, int((analysis['right_db'] + 80) / 4))
            else:
                left_bars = '▓' * min(20, int((analysis['left_db'] + 80) / 4))
                right_bars = '█' * min(20, int((analysis['right_db'] + 80) / 4))
            
            print(f"\nVolume Level Comparison:")
            print(f"L: {left_bars:<20} ({analysis['left_db']:.1f} dBFS)")
            print(f"R: {right_bars:<20} ({analysis['right_db']:.1f} dBFS)")
            
            print(f"\n{Colors.BOLD}Status:{Colors.RESET} {analysis['active_channels']}")
            print(f"{Colors.BOLD}Auto Recommendation:{Colors.RESET} {analysis['recommendation']}")
            
            print(f"\n{Colors.BOLD}Manual Options:{Colors.RESET}")
            if analysis['recommendation'] == 'use_left_as_source':
                print(f"{Colors.GREEN}Recommended: --use-left-channel (L to stereo){Colors.RESET}")
                print(f"Alternative: --use-right-channel (R to stereo)")
            elif analysis['recommendation'] == 'use_right_as_source':
                print(f"{Colors.GREEN}Recommended: --use-right-channel (R to stereo){Colors.RESET}")
                print(f"Alternative: --use-left-channel (L to stereo)")
            elif analysis['recommendation'] == 'keep_as_is':
                print(f"{Colors.BLUE}No channel merge needed (both channels have content){Colors.RESET}")
                print(f"Option: --use-left-channel or --use-right-channel to force a merge")
            
        elif audio.channels == 1:
            print(f"\n{Colors.BLUE}Single-channel file - will be automatically converted to stereo{Colors.RESET}")
        else:
            print(f"\n{Colors.YELLOW}Multi-channel file ({audio.channels} channels) - will be converted to stereo{Colors.RESET}")
            
    except Exception as e:
        print(f"Error analyzing file: {e}")

def create_unified_parser() -> argparse.ArgumentParser:
    """Creates a unified command line parser with all arguments."""
    parser = argparse.ArgumentParser(description="Professional Audio Processor Unified v5.8 with Enhanced Peak Control & Improved Truncate Silence")
    
    # Input and output files
    parser.add_argument('input', help='Input audio file or pattern for batch mode')
    parser.add_argument('output', nargs='?', help='Output audio file (optional in batch mode)')
    
    # Processing presets and modes
    parser.add_argument('--preset', choices=['sermon_ultra', 'broadcast_pro', 'podcast_pro', 'music_mastering'],
                       default='sermon_ultra', help='Select a quality preset (default: sermon_ultra)')
    parser.add_argument('--speed-preset', choices=['quality', 'normal', 'fast', 'ultra_fast'],
                       default='normal', help='Processing speed preset')
    
    # Performance options
    parser.add_argument('--threads', type=int, help='Number of processing threads')
    parser.add_argument('--chunk-size', type=int, help='Chunk size in milliseconds')
    parser.add_argument('--debug-speed', action='store_true', help='Display detailed timing information')
    
    # Channel processing
    parser.add_argument('--use-left-channel', action='store_true', help='Use the left channel as the source')
    parser.add_argument('--use-right-channel', action='store_true', help='Use the right channel as the source')
    parser.add_argument('--channel-analysis', action='store_true', help='Analyze channels only')
    
    # Audio processing options
    parser.add_argument('--enable-eq', action='store_true', help='Enable EQ processing')
    parser.add_argument('--disable-eq', action='store_true', help='Disable EQ processing')
    parser.add_argument('--gain', type=float, help='Set gain in dB')
    parser.add_argument('--no-normalize', action='store_true', help='Disable normalization')
    parser.add_argument('--no-compression', action='store_true', help='Disable compression')
    parser.add_argument('--no-noise-reduction', action='store_true', help='Disable noise reduction')
    parser.add_argument('--no-limiter', action='store_true', help='Disable the final limiter')
    parser.add_argument('--no-stereo-enhancement', action='store_true', help='Disable stereo enhancement')
    parser.add_argument('--no-silence-processing', action='store_true', help='Disable silence processing')
    
    # ✅ Silence processing control options (NEW)
    parser.add_argument('--no-trim-leading', action='store_true', help='Disable leading silence trimming')
    parser.add_argument('--no-trim-trailing', action='store_true', help='Disable trailing silence trimming')
    parser.add_argument('--trim-leading', action='store_true', help='Force enable leading silence trimming')
    parser.add_argument('--trim-trailing', action='store_true', help='Force enable trailing silence trimming')

    # Peak control options (NEW)
    parser.add_argument('--disable-peak-control', action='store_true', help='Disable pre-normalization peak control')
    parser.add_argument('--peak-threshold', type=float, default=-1.0, help='Peak detection threshold in dBFS')
    
    # Output format options
    parser.add_argument('--format', choices=['mp3', 'wav', 'flac', 'ogg'], help='Output file format')
    parser.add_argument('--bitrate', type=str, help='Output bitrate (e.g., 192k)')
    parser.add_argument('--sample-rate', type=int, help='Output sample rate')
    
    # Other options
    parser.add_argument('--batch', action='store_true', help='Enable batch processing')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files')
    parser.add_argument('--quiet', action='store_true', help='Reduce console output')
    parser.add_argument('--analyze', action='store_true', help='Analyze audio before processing')
    parser.add_argument('--skip-analysis', action='store_true', help='Skip audio analysis')
    parser.add_argument('--fast-mode', action='store_true', help='Enable fast processing mode')
    parser.add_argument('--ultra-fast', action='store_true', help='Enable ultra-fast processing')
    parser.add_argument('--load-config', type=str, help='Load settings from a JSON file')
    parser.add_argument('--save-config', type=str, help='Save settings to a JSON file')
    
    return parser

async def main():
    """Main application entry point"""
    
    parser = create_unified_parser()
    args = parser.parse_args()
    
    # Configure speed settings
    speed_config = SPEED_PRESETS[args.speed_preset].copy()
    
    # Override with command line options
    if args.fast_mode:
        speed_config = SPEED_PRESETS['fast'].copy()
        print(f"{Colors.YELLOW}Fast mode enabled - reducing quality for speed{Colors.RESET}")
    
    if args.ultra_fast:
        speed_config = SPEED_PRESETS['ultra_fast'].copy()
        print(f"{Colors.YELLOW}Ultra-fast mode enabled - minimal processing{Colors.RESET}")
    
    if args.threads:
        speed_config['max_threads'] = min(args.threads, cpu_count())
    
    if args.chunk_size:
        speed_config['chunk_size_ms'] = args.chunk_size
    
    if args.skip_analysis:
        speed_config['enable_analysis'] = False
        print(f"{Colors.CYAN}Disabling analysis for faster processing{Colors.RESET}")
    
    # Configure global settings
    global THREAD_POOL_SIZE, CHUNK_SIZE_MS
    THREAD_POOL_SIZE = speed_config['max_threads']
    CHUNK_SIZE_MS = speed_config['chunk_size_ms']
    
    # Header
    if not args.quiet:
        print(f"\n{Colors.BOLD}{Colors.BLUE}Professional Audio Processor Unified v5.8 with Enhanced Peak Control & Improved Truncate Silence{Colors.RESET}")
        print(f"{Colors.BLUE}{'=' * 85}{Colors.RESET}")
        print(f"Features: Intelligent Channel Detection & Stereo Conversion | Peak Control: {'Enabled' if not args.disable_peak_control else 'Disabled'} | EQ: {'Enabled' if args.enable_eq else 'Disabled'}")
        print()
    
    # Channel analysis only mode
    if args.channel_analysis:
        if not os.path.exists(args.input):
            print(f"Error: File not found: {args.input}")
            return
        analyze_channels_only(args.input)
        return
    
    # Load configuration if specified
    config_settings = {}
    if args.load_config and os.path.exists(args.load_config):
        try:
            with open(args.load_config, 'r', encoding='utf-8') as f:
                config_settings = json.load(f)
            print(f"Loaded settings from: {args.load_config}")
        except Exception as e:
            print(f"Error loading config file: {e}")
            return
    
    # Get base settings from preset
    preset_settings = QUALITY_PRESETS.get(args.preset, QUALITY_PRESETS['sermon_ultra']).copy()
    preset_settings.update(config_settings)
    
    # Apply command line overrides
    if args.enable_eq:
        preset_settings['eq_enabled'] = True
    elif args.disable_eq:
        preset_settings['eq_enabled'] = False
    
    if args.gain is not None:
        preset_settings['gain_db'] = args.gain
    
    if args.no_normalize:
        preset_settings['normalize'] = False
    
    if args.no_compression:
        preset_settings['compression'] = None
    
    if args.no_noise_reduction:
        preset_settings['noise_reduction'] = None
    
    if args.no_limiter:
        preset_settings['limiter'] = None
    
    if args.no_stereo_enhancement:
        preset_settings['stereo_enhancement'] = False
    
    if args.no_silence_processing:
        preset_settings['silence_processing'] = None
    
    # ✅ Handle leading/trailing silence options (NEW)
    if args.no_trim_leading or args.no_trim_trailing or args.trim_leading or args.trim_trailing:
        if 'silence_processing' not in preset_settings:
            preset_settings['silence_processing'] = {}
        
        if args.no_trim_leading:
            preset_settings['silence_processing']['process_leading'] = False
            print(f"{Colors.CYAN}Leading silence trimming disabled{Colors.RESET}")
        elif args.trim_leading:
            preset_settings['silence_processing']['process_leading'] = True
            print(f"{Colors.CYAN}Leading silence trimming enabled{Colors.RESET}")
            
        if args.no_trim_trailing:
            preset_settings['silence_processing']['process_trailing'] = False
            print(f"{Colors.CYAN}Trailing silence trimming disabled{Colors.RESET}")
        elif args.trim_trailing:
            preset_settings['silence_processing']['process_trailing'] = True
            print(f"{Colors.CYAN}Trailing silence trimming enabled{Colors.RESET}")
        
    # Peak control settings (NEW)
    if args.disable_peak_control:
        preset_settings['peak_control'] = {'enabled': False}
    else:
        if 'peak_control' not in preset_settings:
            preset_settings['peak_control'] = {'enabled': True}
        preset_settings['peak_control']['threshold_db'] = args.peak_threshold
    
    # Apply output format settings
    if args.format:
        preset_settings['output_format'] = args.format
    if args.bitrate:
        preset_settings['bitrate'] = args.bitrate
    if args.sample_rate:
        preset_settings['sample_rate'] = args.sample_rate
    # Apply output format settings
    if args.format:
        preset_settings['output_format'] = args.format
    if args.bitrate:
        preset_settings['bitrate'] = args.bitrate
    if args.sample_rate:
        preset_settings['sample_rate'] = args.sample_rate        
    
    # Save configuration if requested
    if args.save_config:
        try:
            with open(args.save_config, 'w', encoding='utf-8') as f:
                json.dump(preset_settings, f, indent=2, ensure_ascii=False)
            print(f"Saved settings to: {args.save_config}")
            return
        except Exception as e:
            print(f"Error saving configuration: {e}")
            return
    
    # Display processing settings
    if not args.quiet:
        print(f"Preset: {Colors.CYAN}{preset_settings['name']}{Colors.RESET}")
        print(f"Peak Control: {'Enabled' if preset_settings.get('peak_control', {}).get('enabled', True) else 'Disabled'}")
        print(f"EQ: {'Enabled' if preset_settings.get('eq_enabled', False) else 'Disabled'}")
        print(f"Enhanced Truncate Silence: {'Enabled' if preset_settings.get('silence_processing', {}).get('use_fast_detection', False) else 'Disabled'}")
        print(f"Threads: {THREAD_POOL_SIZE} | Chunk Size: {CHUNK_SIZE_MS}ms")
    
    # ✅ Show leading/trailing silence settings
        if preset_settings.get('silence_processing'):
            silence_config = preset_settings['silence_processing']
            process_leading = silence_config.get('process_leading', False)
            process_trailing = silence_config.get('process_trailing', True)
            print(f"Leading Silence: {'Enabled' if process_leading else 'Disabled'} | Trailing Silence: {'Enabled' if process_trailing else 'Disabled'}")
        print(f"Threads: {THREAD_POOL_SIZE} | Chunk Size: {CHUNK_SIZE_MS}ms")
     
    # Process files
    if args.batch:
        await batch_process_files([args.input], preset_settings, args)
    else:
        # Single file processing
        if not args.output:
            input_path = Path(args.input)
            if args.format:
                output_ext = f".{args.format}"
            else:
                output_ext = input_path.suffix
            args.output = str(input_path.parent / f"{input_path.stem}_processed{output_ext}")
        
        # Check if output exists
        if os.path.exists(args.output) and not args.overwrite:
            if input("Output file exists. Overwrite? (y/N): ").lower() != 'y':
                return
        
        # Process single file
        processor = UnifiedAudioProcessor(speed_config)
        success = await processor.process_audio_async(args.input, args.output, preset_settings, args)
        
        if success and not args.quiet:
            print(f"\n{Colors.GREEN}Processing completed successfully!{Colors.RESET}")
            print(f"Output file: {Colors.CYAN}{args.output}{Colors.RESET}")
        elif not success:
            print(f"\n{Colors.RED}Processing failed{Colors.RESET}")

if __name__ == "__main__":
    try:
        # Check Python version
        if sys.version_info < (3, 7):
            print("Error: Python 3.7 or newer is required.")
            sys.exit(1)
        
        # Run async main
        asyncio.run(main())
        
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Processing cancelled by user{Colors.RESET}")
    except Exception as e:
        print(f"\n{Colors.RED}An unexpected error occurred: {e}{Colors.RESET}")
        if "--debug" in sys.argv:
            import traceback
            traceback.print_exc()

# Usage Examples:
# python audio_processor_pro-v5.8.py input.mp3 output.mp3 --preset sermon_ultra --gain 12.0
# python audio_processor_pro-v5.8.py input.mp3 output.mp3 --preset sermon_ultra --enable-eq --debug-speed
# python audio_processor_pro-v5.8.py input.mp3 --channel-analysis
# python audio_processor_pro-v5.8.py "*.mp3" --batch --preset podcast_pro --overwrite
