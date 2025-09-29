#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio Processor GUI v1.1 - FIXED VERSION
========================================

Enhanced GUI interface for audio_processor_pro-v5.8.py
Fixed: Custom preset handling and mouse wheel scrolling issues

Author: Enhanced Audio Processing Pro
License: MIT

"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, simpledialog
import json
import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
import subprocess
import threading
import queue
from pathlib import Path
from typing import Dict, List, Any
import tempfile
import shutil
from datetime import datetime
import copy

# Import the original script's constants and presets
# This assumes the script is in the same directory
try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from audio_processor import QUALITY_PRESETS, SPEED_PRESETS
except ImportError:
    # Fallback presets if import fails
    QUALITY_PRESETS = {
        'sermon_ultra': {'name': 'Sermon Ultra', 'description': 'Premium sermon processing'},
        'broadcast_pro': {'name': 'Broadcast Pro', 'description': 'Professional broadcast quality'},
        'podcast_pro': {'name': 'Podcast Pro', 'description': 'High-quality podcast processing'},
        'music_mastering': {'name': 'Music Mastering', 'description': 'Professional music mastering'}
    }
    SPEED_PRESETS = {
        'quality': {'name': 'Quality'},
        'normal': {'name': 'Normal'},
        'fast': {'name': 'Fast'},
        'ultra_fast': {'name': 'Ultra Fast'}
    }

# Default module configurations
DEFAULT_MODULE_CONFIGS = {
    'normalize': {
        'enabled': True,
        'headroom_db': 0.1,
        'preserve_dynamics': True
    },
    'peak_control': {
        'enabled': True,
        'threshold_db': -1.0,
        'pre_normalization': True,
        'safety_limiter': True,
        'release_ms': 5.0,
        'lookahead_ms': 1.0
    },
    'volume_boost': {
        'enabled': False,
        'target_lufs': -16.0,
        'max_peak_db': -0.3,
        'adaptive_gain': True,
        'preserve_dynamics': True,
        'fixed_gain': 6.0
    },
    'compression': {
        'enabled': False,
        'threshold': -20.0,
        'ratio': 3.0,
        'attack': 5,
        'release': 250,
        'knee': 2.0,
        'makeup_gain': 0.0,
        'parallel': {
            'enabled': True,
            'mix': 50.0,
            'chunk_size': 12000,
            'threads': 6,
            'overlap_ms': 25
        }
    },
    'noise_reduction': {
        'enabled': False,
        'strength': 0.8,
        'preserve_speech': True,
        'sensitivity': 0.5,
        'frequency_masking': True,
        'temporal_masking': True,
        'chunk_duration': 30.0
    },
    'silence_processing': {
        'enabled': True,
        'audacity_style': True,
        'threshold_db': -35.0,
        'min_silence_to_process': 750,
        'max_internal_silence_ms': 600,
        'trim_silence': True,
        'process_leading': True,    # ✅ Default: ตัดเสียงเงียบช่วงเริ่มต้น
        'process_trailing': True,    # ✅ Default: ตัดเสียงเงียบช่วงท้าย        
        'compress_internal': True,
        'sentence_break_detection': True,
        'sentence_break_keep_ratio': 0.8,
        'padding': 50,
        'edge_protection_sec': 0.5,
        'use_fast_detection': True,
        'chunk_processing': True,
        'fade_duration_ms': 50,
        'debug_mode': False
    },
    'eq': {
        'enabled': False,
        'bands': {
            'bass': {'freq': 120, 'gain': 0.0, 'q': 1.0, 'type': 'peaking'},
            'low_mid': {'freq': 400, 'gain': 0.0, 'q': 1.3, 'type': 'peaking'},
            'mid': {'freq': 1000, 'gain': 0.0, 'q': 1.0, 'type': 'peaking'},
            'high_mid': {'freq': 2800, 'gain': 0.0, 'q': 1.2, 'type': 'peaking'},
            'presence': {'freq': 5000, 'gain': 0.0, 'q': 0.8, 'type': 'peaking'},
            'brilliance': {'freq': 10000, 'gain': 0.0, 'q': 0.7, 'type': 'peaking'}
        },
        'high_pass': {
            'enabled': False,
            'frequency': 80.0,
            'slope': 12
        },
        'low_pass': {
            'enabled': False,
            'frequency': 15000.0,
            'slope': 12
        }
    },
    'limiter': {
        'enabled': True,
        'threshold': -1.0,
        'release': 50,
        'knee': 0.1,
        'lookahead': 5.0,
        'isr': 4
    }
}

class ConfigManager:
    """Manages configuration files and presets"""
    
    def __init__(self):
        self.config_dir = Path.home() / ".audio_processor_gui"
        self.config_dir.mkdir(exist_ok=True)
        
        self.presets_file = self.config_dir / "presets.json"
        self.custom_presets_file = self.config_dir / "custom_presets.json"
        self.app_settings_file = self.config_dir / "app_settings.json"
        
        self.default_presets = QUALITY_PRESETS.copy()
        self.custom_presets = self.load_custom_presets()
        self.app_settings = self.load_app_settings()
    
    def load_custom_presets(self) -> Dict:
        """Load custom presets from file"""
        if self.custom_presets_file.exists():
            try:
                with open(self.custom_presets_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def save_custom_presets(self):
        """Save custom presets to file"""
        try:
            with open(self.custom_presets_file, 'w', encoding='utf-8') as f:
                json.dump(self.custom_presets, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving custom presets: {e}")
    
    def load_app_settings(self) -> Dict:
        """Load application settings"""
        if self.app_settings_file.exists():
            try:
                with open(self.app_settings_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {
            'last_input_dir': str(Path.home()),
            'last_output_dir': str(Path.home()),
            'window_geometry': '1200x800'
        }
    
    def save_app_settings(self):
        """Save application settings"""
        try:
            with open(self.app_settings_file, 'w', encoding='utf-8') as f:
                json.dump(self.app_settings, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving app settings: {e}")
    
    def get_all_presets(self) -> Dict:
        """Get combined default and custom presets"""
        all_presets = self.default_presets.copy()
        all_presets.update(self.custom_presets)
        return all_presets


class ModuleConfigWindow:
    """Enhanced configuration window for individual modules"""
    
    def __init__(self, parent, module_name: str, config: Dict, callback=None):
        self.parent = parent
        self.module_name = module_name
        self.original_config = copy.deepcopy(config)
        self.config = copy.deepcopy(config)
        self.callback = callback
        self.result = None
        self.config_vars = {}
        self.mouse_bound_canvas = None  # Track canvas for mouse wheel binding
        
        self.window = tk.Toplevel(parent)
        self.window.title(f"Configure {module_name.replace('_', ' ').title()}")
        self.window.geometry("700x600")
        self.window.transient(parent)
        self.window.grab_set()
        
        # Make window resizable
        self.window.rowconfigure(0, weight=1)
        self.window.columnconfigure(0, weight=1)
        
        self.create_widgets()
        self.center_window()
        
        # Bind close event to cleanup
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
    
    def on_close(self):
        """Handle window close - cleanup mouse wheel binding"""
        if self.mouse_bound_canvas:
            try:
                self.mouse_bound_canvas.unbind_all("<MouseWheel>")
            except:
                pass
        self.window.destroy()
    
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_frame.rowconfigure(1, weight=1)
        main_frame.columnconfigure(0, weight=1)
        
        # Title and description
        title_frame = ttk.Frame(main_frame)
        title_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        title_label = ttk.Label(title_frame, text=f"{self.module_name.replace('_', ' ').title()} Configuration", 
                               font=("Arial", 14, "bold"))
        title_label.grid(row=0, column=0, sticky=tk.W)
        
        # Add description based on module type
        descriptions = {
            'normalize': 'Normalize audio levels to optimal range for consistent volume',
            'peak_control': 'Advanced peak detection and limiting before normalization',
            'volume_boost': 'Intelligent volume boosting with LUFS targeting',
            'compression': 'Dynamic range compression with parallel processing',
            'noise_reduction': 'AI-powered noise reduction with speech preservation',
            'silence_processing': 'Enhanced Audacity-style silence truncation',
            'eq': 'Multi-band parametric equalizer with filters',
            'limiter': 'Final safety limiter with lookahead',
        }
        
        desc_text = descriptions.get(self.module_name, 'Configure module parameters')
        desc_label = ttk.Label(title_frame, text=desc_text, foreground="gray")
        desc_label.grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        
        # Configuration area with scrollbar
        config_frame = ttk.Frame(main_frame)
        config_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        config_frame.rowconfigure(0, weight=1)
        config_frame.columnconfigure(0, weight=1)
        
        # Canvas and scrollbar
        canvas = tk.Canvas(config_frame)
        scrollbar = ttk.Scrollbar(config_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Create module-specific configuration
        self.create_module_config(scrollable_frame)
        
        canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, pady=(10, 0))
        
        ttk.Button(button_frame, text="OK", command=self.ok_clicked).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Cancel", command=self.cancel_clicked).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Reset to Default", command=self.reset_clicked).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Load Preset", command=self.load_preset_clicked).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Preset", command=self.save_preset_clicked).pack(side=tk.LEFT, padx=5)
        
        # FIXED: Safe mouse wheel binding
        def _on_mousewheel(event):
            try:
                if canvas.winfo_exists():
                    canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            except tk.TclError:
                pass  # Canvas no longer exists, ignore
        
        # Store reference and bind
        self.mouse_bound_canvas = canvas
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
    
    def create_module_config(self, parent):
        """Create module-specific configuration interface"""
        
        if self.module_name == 'eq':
            self.create_eq_config(parent)
        elif self.module_name == 'compression':
            self.create_compression_config(parent)
        elif self.module_name == 'silence_processing':
            self.create_silence_config(parent)
        elif self.module_name == 'noise_reduction':
            self.create_noise_reduction_config(parent)
        elif self.module_name == 'volume_boost':
            self.create_volume_boost_config(parent)
        elif self.module_name == 'peak_control':
            self.create_peak_control_config(parent)
        elif self.module_name == 'limiter':
            self.create_limiter_config(parent)
        else:
            # Generic config for simple modules
            self.create_generic_config(parent)
    
    def create_eq_config(self, parent):
        """Create EQ-specific configuration interface"""
        row = 0
        
        # Main enable
        enabled_var = tk.BooleanVar(value=self.config.get('enabled', False))
        self.config_vars['enabled'] = enabled_var
        ttk.Checkbutton(parent, text="Enable EQ", variable=enabled_var).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=5)
        row += 1
        
        # EQ Bands
        bands_frame = ttk.LabelFrame(parent, text="EQ Bands", padding="10")
        bands_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5, padx=5)
        row += 1
        
        bands = self.config.get('bands', {})
        for i, (band_name, band_config) in enumerate(bands.items()):
            band_frame = ttk.Frame(bands_frame)
            band_frame.grid(row=i, column=0, sticky=(tk.W, tk.E), pady=2)
            
            # Band label
            ttk.Label(band_frame, text=f"{band_name.replace('_', ' ').title()}:", width=12).grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
            
            # Frequency
            ttk.Label(band_frame, text="Freq:").grid(row=0, column=1, sticky=tk.W)
            freq_var = tk.StringVar(value=str(band_config.get('freq', 1000)))
            self.config_vars[f'bands.{band_name}.freq'] = freq_var
            ttk.Entry(band_frame, textvariable=freq_var, width=8).grid(row=0, column=2, padx=(2, 10))
            
            # Gain
            ttk.Label(band_frame, text="Gain:").grid(row=0, column=3, sticky=tk.W)
            gain_var = tk.StringVar(value=str(band_config.get('gain', 0.0)))
            self.config_vars[f'bands.{band_name}.gain'] = gain_var
            ttk.Entry(band_frame, textvariable=gain_var, width=8).grid(row=0, column=4, padx=(2, 10))
            
            # Q
            ttk.Label(band_frame, text="Q:").grid(row=0, column=5, sticky=tk.W)
            q_var = tk.StringVar(value=str(band_config.get('q', 1.0)))
            self.config_vars[f'bands.{band_name}.q'] = q_var
            ttk.Entry(band_frame, textvariable=q_var, width=8).grid(row=0, column=6, padx=2)
        
        # High Pass Filter
        hp_frame = ttk.LabelFrame(parent, text="High Pass Filter", padding="10")
        hp_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5, padx=5)
        row += 1
        
        hp_enabled_var = tk.BooleanVar(value=self.config.get('high_pass', {}).get('enabled', False))
        self.config_vars['high_pass.enabled'] = hp_enabled_var
        ttk.Checkbutton(hp_frame, text="Enable", variable=hp_enabled_var).grid(row=0, column=0, sticky=tk.W)
        
        ttk.Label(hp_frame, text="Frequency (Hz):").grid(row=0, column=1, sticky=tk.W, padx=(20, 5))
        hp_freq_var = tk.StringVar(value=str(self.config.get('high_pass', {}).get('frequency', 80.0)))
        self.config_vars['high_pass.frequency'] = hp_freq_var
        ttk.Entry(hp_frame, textvariable=hp_freq_var, width=10).grid(row=0, column=2, padx=5)
        
        # Low Pass Filter
        lp_frame = ttk.LabelFrame(parent, text="Low Pass Filter", padding="10")
        lp_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5, padx=5)
    
    def create_compression_config(self, parent):
        """Create compression-specific configuration interface"""
        row = 0
        
        # Enable
        enabled_var = tk.BooleanVar(value=self.config.get('enabled', True))
        self.config_vars['enabled'] = enabled_var
        ttk.Checkbutton(parent, text="Enable Compression", variable=enabled_var).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=5)
        row += 1
        
        # Basic parameters
        basic_frame = ttk.LabelFrame(parent, text="Basic Parameters", padding="10")
        basic_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5, padx=5)
        row += 1
        
        params = [
            ('threshold', 'Threshold (dB)', -20.0),
            ('ratio', 'Ratio', 3.0),
            ('attack', 'Attack (ms)', 5),
            ('release', 'Release (ms)', 250),
            ('knee', 'Knee (dB)', 2.0),
            ('makeup_gain', 'Makeup Gain (dB)', 0.0)
        ]
        
        for i, (key, label, default) in enumerate(params):
            ttk.Label(basic_frame, text=f"{label}:").grid(row=i//2, column=(i%2)*2, sticky=tk.W, padx=(0, 5), pady=2)
            var = tk.StringVar(value=str(self.config.get(key, default)))
            self.config_vars[key] = var
            ttk.Entry(basic_frame, textvariable=var, width=12).grid(row=i//2, column=(i%2)*2+1, padx=(0, 20), pady=2)
        
        # Parallel processing
        parallel_frame = ttk.LabelFrame(parent, text="Parallel Processing", padding="10")
        parallel_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5, padx=5)
        
        parallel_config = self.config.get('parallel', {})
        
        parallel_enabled_var = tk.BooleanVar(value=parallel_config.get('enabled', True))
        self.config_vars['parallel.enabled'] = parallel_enabled_var
        ttk.Checkbutton(parallel_frame, text="Enable Parallel Processing", variable=parallel_enabled_var).grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        parallel_params = [
            ('mix', 'Mix %', 50.0),
            ('chunk_size', 'Chunk Size (ms)', 12000),
            ('threads', 'Threads', 6),
            ('overlap_ms', 'Overlap (ms)', 25)
        ]
        
        for i, (key, label, default) in enumerate(parallel_params):
            ttk.Label(parallel_frame, text=f"{label}:").grid(row=1+i//2, column=(i%2)*2, sticky=tk.W, padx=(0, 5), pady=2)
            var = tk.StringVar(value=str(parallel_config.get(key, default)))
            self.config_vars[f'parallel.{key}'] = var
            ttk.Entry(parallel_frame, textvariable=var, width=12).grid(row=1+i//2, column=(i%2)*2+1, padx=(0, 20), pady=2)
    
    def create_silence_config(self, parent):
        """Create silence processing configuration interface - COMPLETE VERSION"""
        row = 0
        
        # Enable
        enabled_var = tk.BooleanVar(value=self.config.get('enabled', True))
        self.config_vars['enabled'] = enabled_var
        enable_cb = ttk.Checkbutton(parent, text="Enable Silence Processing", variable=enabled_var)
        enable_cb.grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=5)
        row += 1
        
        # Main parameters
        main_frame = ttk.LabelFrame(parent, text="Main Parameters", padding="10")
        main_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5, padx=5)
        main_frame.columnconfigure(1, weight=1)
        main_frame.columnconfigure(3, weight=1)
        row += 1
        
        main_params = [
            ('threshold_db', 'Threshold (dB)', -25.0),
            ('min_silence_to_process', 'Min Silence Duration (ms)', 750),
            ('max_internal_silence_ms', 'Max Keep Silence (ms)', 600),
            ('padding', 'Padding (ms)', 50)
        ]
        
        for i, (key, label, default) in enumerate(main_params):
            param_row = i // 2
            param_col = (i % 2) * 2
            
            ttk.Label(main_frame, text=f"{label}:").grid(
                row=param_row, column=param_col, sticky=tk.W, padx=(0, 5), pady=2)
            var = tk.StringVar(value=str(self.config.get(key, default)))
            self.config_vars[key] = var
            ttk.Entry(main_frame, textvariable=var, width=12).grid(
                row=param_row, column=param_col+1, padx=(0, 20), pady=2, sticky=tk.W)
        
        # Silence Type Controls
        control_frame = ttk.LabelFrame(parent, text="Silence Types to Process", padding="10")
        control_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5, padx=5)
        control_frame.columnconfigure(0, weight=1)
        row += 1
        
        # 1. Leading/Trailing Silence Toggle
        trim_var = tk.BooleanVar(value=self.config.get('trim_silence', True))
        self.config_vars['trim_silence'] = trim_var
        trim_cb = ttk.Checkbutton(control_frame, text="Process Leading/Trailing Silence", 
                                 variable=trim_var)
        trim_cb.grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        # Sub-controls for leading/trailing
        sub_frame = ttk.Frame(control_frame)
        sub_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), padx=20, pady=(0, 10))
        sub_frame.columnconfigure(0, weight=1)
        
        # Leading silence control
        leading_var = tk.BooleanVar(value=self.config.get('process_leading', False))
        self.config_vars['process_leading'] = leading_var
        leading_cb = ttk.Checkbutton(sub_frame, text="Leading Silence (start of audio)", 
                                    variable=leading_var)
        leading_cb.grid(row=0, column=0, sticky=tk.W, pady=2)
        
        # Warning label
        warning_label = ttk.Label(sub_frame, 
                                 text="⚠️ Warning: May cut important content", 
                                 foreground="orange")
        warning_label.grid(row=1, column=0, sticky=tk.W, padx=20, pady=(0, 5))
        
        # Trailing silence control
        trailing_var = tk.BooleanVar(value=self.config.get('process_trailing', True))
        self.config_vars['process_trailing'] = trailing_var
        trailing_cb = ttk.Checkbutton(sub_frame, text="Trailing Silence (end of audio)", 
                                     variable=trailing_var)
        trailing_cb.grid(row=2, column=0, sticky=tk.W, pady=2)
        
        # 2. Internal Silence Control
        internal_var = tk.BooleanVar(value=self.config.get('compress_internal', True))
        self.config_vars['compress_internal'] = internal_var
        internal_cb = ttk.Checkbutton(control_frame, text="Compress Internal Silence (within audio)", 
                                     variable=internal_var)
        internal_cb.grid(row=2, column=0, sticky=tk.W, pady=(10, 0))
        
        # Enable/disable sub-controls based on trim_silence
        def toggle_sub_controls():
            state = tk.NORMAL if trim_var.get() else tk.DISABLED
            leading_cb.config(state=state)
            trailing_cb.config(state=state)
            color = "orange" if trim_var.get() else "gray"
            warning_label.config(foreground=color)
        
        trim_var.trace_add("write", lambda *args: toggle_sub_controls())
        toggle_sub_controls()  # Set initial state
        
        # Advanced Options
        options_frame = ttk.LabelFrame(parent, text="Advanced Options", padding="10")
        options_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5, padx=5)
        options_frame.columnconfigure(0, weight=1)
        options_frame.columnconfigure(1, weight=1)
        row += 1
        
        options = [
            ('sentence_break_detection', 'Smart Sentence Break Detection', True),
            ('audacity_style', 'Audacity-Style Processing', True),
            ('use_fast_detection', 'Use Fast Detection', True),
            ('chunk_processing', 'Chunk Processing for Large Files', True)
        ]
        
        for i, (key, label, default) in enumerate(options):
            option_row = i // 2
            option_col = i % 2
            
            var = tk.BooleanVar(value=self.config.get(key, default))
            self.config_vars[key] = var
            ttk.Checkbutton(options_frame, text=label, variable=var).grid(
                row=option_row, column=option_col, sticky=tk.W, padx=10, pady=2)
        
        # Advanced Parameters
        advanced_frame = ttk.LabelFrame(parent, text="Advanced Parameters", padding="10")
        advanced_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5, padx=5)
        advanced_frame.columnconfigure(1, weight=1)
        advanced_frame.columnconfigure(3, weight=1)
        
        advanced_params = [
            ('sentence_break_keep_ratio', 'Sentence Break Keep Ratio', 0.6),
            ('edge_protection_sec', 'Edge Protection (sec)', 0.2),
            ('fade_duration_ms', 'Fade Duration (ms)', 3)
        ]
        
        for i, (key, label, default) in enumerate(advanced_params):
            param_row = i // 2
            param_col = (i % 2) * 2
            
            ttk.Label(advanced_frame, text=f"{label}:").grid(
                row=param_row, column=param_col, sticky=tk.W, padx=(0, 5), pady=2)
            var = tk.StringVar(value=str(self.config.get(key, default)))
            self.config_vars[key] = var
            ttk.Entry(advanced_frame, textvariable=var, width=12).grid(
                row=param_row, column=param_col+1, padx=(0, 20), pady=2, sticky=tk.W)
            
    def create_noise_reduction_config(self, parent):
        """Create noise reduction configuration interface"""
        row = 0
        
        # Enable
        enabled_var = tk.BooleanVar(value=self.config.get('enabled', False))
        self.config_vars['enabled'] = enabled_var
        ttk.Checkbutton(parent, text="Enable Noise Reduction", variable=enabled_var).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=5)
        row += 1
        
        # Warning label
        warning_frame = ttk.Frame(parent)
        warning_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        warning_label = ttk.Label(warning_frame, text="⚠️ Note: Requires noisereduce library (pip install noisereduce)", 
                                 foreground="orange", font=("Arial", 9))
        warning_label.grid(row=0, column=0, sticky=tk.W)
        row += 1
        
        # Parameters
        params_frame = ttk.LabelFrame(parent, text="Parameters", padding="10")
        params_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5, padx=5)
        
        params = [
            ('strength', 'Noise Reduction Strength (0.0-1.0)', 0.8),
            ('sensitivity', 'Sensitivity (0.0-1.0)', 0.5),
            ('chunk_duration', 'Processing Chunk Duration (sec)', 30.0)
        ]
        
        for i, (key, label, default) in enumerate(params):
            ttk.Label(params_frame, text=f"{label}:").grid(row=i, column=0, sticky=tk.W, padx=(0, 5), pady=2)
            var = tk.StringVar(value=str(self.config.get(key, default)))
            self.config_vars[key] = var
            ttk.Entry(params_frame, textvariable=var, width=12).grid(row=i, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Options
        options = [
            ('preserve_speech', 'Preserve Speech Quality', True),
            ('frequency_masking', 'Frequency Masking', True),
            ('temporal_masking', 'Temporal Masking', True)
        ]
        
        for i, (key, label, default) in enumerate(options):
            var = tk.BooleanVar(value=self.config.get(key, default))
            self.config_vars[key] = var
            ttk.Checkbutton(params_frame, text=label, variable=var).grid(row=len(params)+i, column=0, columnspan=2, sticky=tk.W, pady=2)
    
    def create_volume_boost_config(self, parent):
        """Create volume boost configuration interface"""
        row = 0
        
        # Enable
        enabled_var = tk.BooleanVar(value=self.config.get('enabled', True))
        self.config_vars['enabled'] = enabled_var
        ttk.Checkbutton(parent, text="Enable Intelligent Volume Boost", variable=enabled_var).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=5)
        row += 1
        
        # Target parameters
        target_frame = ttk.LabelFrame(parent, text="Target Parameters", padding="10")
        target_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5, padx=5)
        row += 1
        
        target_params = [
            ('target_lufs', 'Target LUFS', -16.0),
            ('max_peak_db', 'Maximum Peak (dBFS)', -0.3),
            ('fixed_gain', 'Fixed Gain (if not adaptive)', 6.0)
        ]
        
        for i, (key, label, default) in enumerate(target_params):
            ttk.Label(target_frame, text=f"{label}:").grid(row=i, column=0, sticky=tk.W, padx=(0, 5), pady=2)
            var = tk.StringVar(value=str(self.config.get(key, default)))
            self.config_vars[key] = var
            ttk.Entry(target_frame, textvariable=var, width=12).grid(row=i, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Options
        options = [
            ('adaptive_gain', 'Use Adaptive Gain Calculation', True),
            ('preserve_dynamics', 'Preserve Dynamic Range', True)
        ]
        
        for i, (key, label, default) in enumerate(options):
            var = tk.BooleanVar(value=self.config.get(key, default))
            self.config_vars[key] = var
            ttk.Checkbutton(target_frame, text=label, variable=var).grid(row=len(target_params)+i, column=0, columnspan=2, sticky=tk.W, pady=2)
    
    def create_peak_control_config(self, parent):
        """Create peak control configuration interface"""
        row = 0
        
        # Enable
        enabled_var = tk.BooleanVar(value=self.config.get('enabled', True))
        self.config_vars['enabled'] = enabled_var
        ttk.Checkbutton(parent, text="Enable Peak Control", variable=enabled_var).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=5)
        row += 1
        
        # Parameters
        params_frame = ttk.LabelFrame(parent, text="Parameters", padding="10")
        params_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5, padx=5)
        
        params = [
            ('threshold_db', 'Detection Threshold (dBFS)', -1.0),
            ('release_ms', 'Release Time (ms)', 5.0),
            ('lookahead_ms', 'Lookahead Time (ms)', 1.0)
        ]
        
        for i, (key, label, default) in enumerate(params):
            ttk.Label(params_frame, text=f"{label}:").grid(row=i, column=0, sticky=tk.W, padx=(0, 5), pady=2)
            var = tk.StringVar(value=str(self.config.get(key, default)))
            self.config_vars[key] = var
            ttk.Entry(params_frame, textvariable=var, width=12).grid(row=i, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Options
        options = [
            ('pre_normalization', 'Apply Before Normalization', True),
            ('safety_limiter', 'Enable Safety Limiter', True)
        ]
        
        for i, (key, label, default) in enumerate(options):
            var = tk.BooleanVar(value=self.config.get(key, default))
            self.config_vars[key] = var
            ttk.Checkbutton(params_frame, text=label, variable=var).grid(row=len(params)+i, column=0, columnspan=2, sticky=tk.W, pady=2)
    
    def create_limiter_config(self, parent):
        """Create limiter configuration interface"""
        row = 0
        
        # Enable
        enabled_var = tk.BooleanVar(value=self.config.get('enabled', True))
        self.config_vars['enabled'] = enabled_var
        ttk.Checkbutton(parent, text="Enable Final Limiter", variable=enabled_var).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=5)
        row += 1
        
        # Parameters
        params_frame = ttk.LabelFrame(parent, text="Parameters", padding="10")
        params_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5, padx=5)
        
        params = [
            ('threshold', 'Threshold (dBFS)', -1.0),
            ('release', 'Release Time (ms)', 50),
            ('knee', 'Knee (dB)', 0.1),
            ('lookahead', 'Lookahead (ms)', 5.0),
            ('isr', 'Internal Sample Rate Multiplier', 4)
        ]
        
        for i, (key, label, default) in enumerate(params):
            ttk.Label(params_frame, text=f"{label}:").grid(row=i//2, column=(i%2)*2, sticky=tk.W, padx=(0, 5), pady=2)
            var = tk.StringVar(value=str(self.config.get(key, default)))
            self.config_vars[key] = var
            ttk.Entry(params_frame, textvariable=var, width=10).grid(row=i//2, column=(i%2)*2+1, padx=(0, 15), pady=2)
  
    def create_generic_config(self, parent):
        """Create generic configuration interface for simple modules"""
        row = 0
        
        # Enable
        enabled_var = tk.BooleanVar(value=self.config.get('enabled', True))
        self.config_vars['enabled'] = enabled_var
        ttk.Checkbutton(parent, text=f"Enable {self.module_name.replace('_', ' ').title()}", 
                       variable=enabled_var).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=5)
        row += 1
        
        # Generic parameters based on config structure
        params_frame = ttk.LabelFrame(parent, text="Parameters", padding="10")
        params_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5, padx=5)
        
        param_row = 0
        for key, value in self.config.items():
            if key == 'enabled':
                continue
            
            ttk.Label(params_frame, text=f"{key.replace('_', ' ').title()}:").grid(
                row=param_row, column=0, sticky=tk.W, padx=(0, 5), pady=2)
            
            if isinstance(value, bool):
                var = tk.BooleanVar(value=value)
                ttk.Checkbutton(params_frame, variable=var).grid(row=param_row, column=1, sticky=tk.W, padx=5, pady=2)
            else:
                var = tk.StringVar(value=str(value))
                ttk.Entry(params_frame, textvariable=var, width=15).grid(row=param_row, column=1, sticky=tk.W, padx=5, pady=2)
            
            self.config_vars[key] = var
            param_row += 1
    
    def get_config_from_vars(self) -> Dict:
        """Convert GUI variables back to config dictionary"""
        result = {}
        
        for full_key, var in self.config_vars.items():
            keys = full_key.split('.')
            current = result
            
            # Navigate/create nested structure
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            
            # Set final value with type conversion
            final_key = keys[-1]
            value = var.get()
            
            # Try to maintain original type
            original_value = self.get_original_value(full_key)
            if isinstance(original_value, bool):
                current[final_key] = value
            elif isinstance(original_value, int):
                try:
                    current[final_key] = int(float(value))
                except ValueError:
                    current[final_key] = original_value
            elif isinstance(original_value, float):
                try:
                    current[final_key] = float(value)
                except ValueError:
                    current[final_key] = original_value
            else:
                current[final_key] = value
        
        return result
    
    def get_original_value(self, full_key):
        """Get original value from config"""
        keys = full_key.split('.')
        current = self.original_config
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        
        return current
    
    def ok_clicked(self):
        """Handle OK button click"""
        self.config = self.get_config_from_vars()
        self.result = "ok"
        if self.callback:
            self.callback(self.config)
        self.on_close()
    
    def cancel_clicked(self):
        """Handle Cancel button click"""
        self.result = "cancel"
        self.on_close()
    
    def reset_clicked(self):
        """Reset to default values"""
        default_config = DEFAULT_MODULE_CONFIGS.get(self.module_name, {})
        
        for full_key, var in self.config_vars.items():
            keys = full_key.split('.')
            current = default_config
            
            for key in keys:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    current = None
                    break
            
            if current is not None:
                if isinstance(var, tk.BooleanVar):
                    var.set(current)
                else:
                    var.set(str(current))
    
    def load_preset_clicked(self):
        """Load preset from file"""
        filename = filedialog.askopenfilename(
            title=f"Load {self.module_name} preset",
            filetypes=[('JSON files', '*.json'), ('All files', '*.*')]
        )
        
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    preset_config = json.load(f)
                
                # Update variables
                for full_key, var in self.config_vars.items():
                    keys = full_key.split('.')
                    current = preset_config
                    
                    for key in keys:
                        if isinstance(current, dict) and key in current:
                            current = current[key]
                        else:
                            current = None
                            break
                    
                    if current is not None:
                        if isinstance(var, tk.BooleanVar):
                            var.set(current)
                        else:
                            var.set(str(current))
                
                messagebox.showinfo("Success", f"Preset loaded from {os.path.basename(filename)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load preset: {str(e)}")
    
    def save_preset_clicked(self):
        """Save preset to file"""
        filename = filedialog.asksaveasfilename(
            title=f"Save {self.module_name} preset",
            defaultextension=".json",
            filetypes=[('JSON files', '*.json'), ('All files', '*.*')],
            initialname=f"{self.module_name}_preset.json"
        )
        
        if filename:
            try:
                config = self.get_config_from_vars()
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                
                messagebox.showinfo("Success", f"Preset saved to {os.path.basename(filename)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save preset: {str(e)}")
    
    def center_window(self):
        """Center window on parent"""
        self.window.update_idletasks()
        x = (self.parent.winfo_x() + self.parent.winfo_width()//2 - 
             self.window.winfo_width()//2)
        y = (self.parent.winfo_y() + self.parent.winfo_height()//2 - 
             self.window.winfo_height()//2)
        self.window.geometry(f"+{x}+{y}")


class AudioProcessorGUI:
    """Main GUI application with enhanced features - FIXED VERSION"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Audio Processor Pro v5.8 - Enhanced GUI Interface")
        
        # Start maximized (cross-platform)
        try:
            import platform
            system = platform.system()
            
            if system == "Windows":
                self.root.state('zoomed')  # Windows maximize
            elif system == "Linux":
                self.root.attributes('-zoomed', True)  # Linux maximize
            elif system == "Darwin":  # macOS
                self.root.attributes('-zoomed', True)  # macOS maximize
            else:
                # Fallback for unknown systems
                self.root.geometry("1400x900")
                self.root.after(100, lambda: self.root.state('zoomed'))
        except:
            # If maximize fails, use large default size
            self.root.geometry("1400x900")
        
        # Configuration manager
        self.config_manager = ConfigManager()
        
        # Processing state
        self.processing = False
        self.process_thread = None
        self.process_queue = queue.Queue()
        
        # Current configuration
        self.current_preset = "sermon_ultra"
        self.current_config = self.config_manager.get_all_presets()[self.current_preset].copy()
        
        # Module configurations with defaults
        self.module_configs = copy.deepcopy(DEFAULT_MODULE_CONFIGS)
        self.module_states = self.get_default_module_states()
        
        # FIXED: Track mouse wheel bindings to prevent errors
        self.mouse_bound_widgets = []
        
        self.create_widgets()
        self.setup_styles()
        self.load_settings()
        
        # Start queue monitoring
        self.root.after(100, self.check_queue)
    
    def get_default_module_states(self) -> Dict[str, bool]:
        """Get default module enable/disable states"""
        return {
            'normalize': True,
            'peak_control': True,
            'volume_boost': False,
            'compression': False,
            'noise_reduction': True,
            'silence_processing': True,
            'eq': False,
            'limiter': True,
        }
    
    def create_widgets(self):
        """Create main GUI widgets"""
        # Create main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.create_main_tab()
        self.create_modules_tab()
        self.create_presets_tab()
        self.create_logs_tab()
    
    def create_main_tab(self):
        """Create main processing tab"""
        main_tab = ttk.Frame(self.notebook)
        self.notebook.add(main_tab, text="Main Processing")
        
        # Input/Output section
        io_frame = ttk.LabelFrame(main_tab, text="Input/Output", padding="10")
        io_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Input selection
        input_frame = ttk.Frame(io_frame)
        input_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(input_frame, text="Input:").pack(side=tk.LEFT)
        self.input_var = tk.StringVar()
        self.input_entry = ttk.Entry(input_frame, textvariable=self.input_var, width=60)
        self.input_entry.pack(side=tk.LEFT, padx=(10, 5), fill=tk.X, expand=True)
        
        ttk.Button(input_frame, text="Browse File", 
                  command=self.browse_input_file).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(input_frame, text="Browse Folder", 
                  command=self.browse_input_folder).pack(side=tk.LEFT)
        
        # Output selection
        output_frame = ttk.Frame(io_frame)
        output_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(output_frame, text="Output:").pack(side=tk.LEFT)
        self.output_var = tk.StringVar()
        self.output_entry = ttk.Entry(output_frame, textvariable=self.output_var, width=60)
        self.output_entry.pack(side=tk.LEFT, padx=(10, 5), fill=tk.X, expand=True)
        
        ttk.Button(output_frame, text="Browse File", 
                  command=self.browse_output_file).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(output_frame, text="Browse Folder", 
                  command=self.browse_output_folder).pack(side=tk.LEFT)
        
        # Processing mode
        mode_frame = ttk.Frame(io_frame)
        mode_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(mode_frame, text="Mode:").pack(side=tk.LEFT)
        self.batch_mode = tk.BooleanVar()
        ttk.Checkbutton(mode_frame, text="Batch Processing", 
                       variable=self.batch_mode).pack(side=tk.LEFT, padx=10)
        
        self.overwrite_var = tk.BooleanVar()
        ttk.Checkbutton(mode_frame, text="Overwrite existing files", 
                       variable=self.overwrite_var).pack(side=tk.LEFT, padx=10)
        
        # Preset and Settings section
        settings_frame = ttk.LabelFrame(main_tab, text="Settings", padding="10")
        settings_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Preset selection
        preset_frame = ttk.Frame(settings_frame)
        preset_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(preset_frame, text="Preset:").pack(side=tk.LEFT)
        self.preset_var = tk.StringVar(value=self.current_preset)
        preset_combo = ttk.Combobox(preset_frame, textvariable=self.preset_var,
                                   values=list(self.config_manager.get_all_presets().keys()),
                                   state="readonly", width=20)
        preset_combo.pack(side=tk.LEFT, padx=10)
        preset_combo.bind('<<ComboboxSelected>>', self.on_preset_changed)
        
        ttk.Button(preset_frame, text="Load Custom", 
                  command=self.load_custom_preset).pack(side=tk.LEFT, padx=5)
        ttk.Button(preset_frame, text="Save Custom", 
                  command=self.save_custom_preset).pack(side=tk.LEFT, padx=5)
        
        # Speed preset
        speed_frame = ttk.Frame(settings_frame)
        speed_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(speed_frame, text="Speed:").pack(side=tk.LEFT)
        self.speed_preset_var = tk.StringVar(value="normal")
        speed_combo = ttk.Combobox(speed_frame, textvariable=self.speed_preset_var,
                                  values=list(SPEED_PRESETS.keys()),
                                  state="readonly", width=15)
        speed_combo.pack(side=tk.LEFT, padx=10)
        
        # Additional options
        options_frame = ttk.Frame(settings_frame)
        options_frame.pack(fill=tk.X, pady=5)
        
        # First row - Gain and Threads
        options_row1 = ttk.Frame(options_frame)
        options_row1.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(options_row1, text="Gain (dB):").pack(side=tk.LEFT)
        self.gain_var = tk.StringVar(value="0.0")
        gain_entry = ttk.Entry(options_row1, textvariable=self.gain_var, width=10)
        gain_entry.pack(side=tk.LEFT, padx=(5, 15))
        
        ttk.Label(options_row1, text="Threads:").pack(side=tk.LEFT)
        self.threads_var = tk.StringVar(value="auto")
        threads_entry = ttk.Entry(options_row1, textvariable=self.threads_var, width=10)
        threads_entry.pack(side=tk.LEFT, padx=(5, 15))
        
        self.debug_var = tk.BooleanVar()
        ttk.Checkbutton(options_row1, text="Debug mode", 
                       variable=self.debug_var).pack(side=tk.LEFT, padx=10)
        
        # Second row - Output Format Options
        options_row2 = ttk.Frame(options_frame)
        options_row2.pack(fill=tk.X, pady=5)
        
        ttk.Label(options_row2, text="Output Format:").pack(side=tk.LEFT)
        self.format_var = tk.StringVar(value="mp3")
        format_combo = ttk.Combobox(options_row2, textvariable=self.format_var,
                                   values=["mp3", "wav", "flac", "ogg"],
                                   state="readonly", width=8)
        format_combo.pack(side=tk.LEFT, padx=(5, 15))
        
        ttk.Label(options_row2, text="Bitrate:").pack(side=tk.LEFT)
        self.bitrate_var = tk.StringVar(value="192k")
        bitrate_combo = ttk.Combobox(options_row2, textvariable=self.bitrate_var,
                                    values=["128k", "160k", "192k", "256k", "320k", "auto"],
                                    state="readonly", width=8)
        bitrate_combo.pack(side=tk.LEFT, padx=(5, 15))
        
        ttk.Label(options_row2, text="Sample Rate:").pack(side=tk.LEFT)
        self.sample_rate_var = tk.StringVar(value="auto")
        sample_rate_combo = ttk.Combobox(options_row2, textvariable=self.sample_rate_var,
                                        values=["auto", "22050", "44100", "48000", "96000"],
                                        state="readonly", width=8)
        sample_rate_combo.pack(side=tk.LEFT, padx=(5, 0))
        
        # Auto-update output file extension when format changes
        def on_format_changed(*args):
            current_output = self.output_var.get()
            if current_output:
                # Change file extension based on format
                output_path = Path(current_output)
                new_extension = f".{self.format_var.get()}"
                new_output = str(output_path.with_suffix(new_extension))
                self.output_var.set(new_output)
        
        self.format_var.trace('w', on_format_changed)
        
        # Progress section
        progress_frame = ttk.LabelFrame(main_tab, text="Processing", padding="10")
        progress_frame.pack(fill=tk.BOTH, expand=True)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, 
                                           mode='determinate', length=400)
        self.progress_bar.pack(fill=tk.X, pady=(0, 10))
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(progress_frame, textvariable=self.status_var)
        status_label.pack(pady=(0, 10))
        
        # Control buttons
        button_frame = ttk.Frame(progress_frame)
        button_frame.pack(fill=tk.X)
        
        self.process_button = ttk.Button(button_frame, text="Start Processing", 
                                        command=self.start_processing, style="Accent.TButton")
        self.process_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_button = ttk.Button(button_frame, text="Stop", 
                                     command=self.stop_processing, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(button_frame, text="Clear", 
                  command=self.clear_fields).pack(side=tk.RIGHT)
    
    def create_modules_tab(self):
        """Create enhanced modules configuration tab with 4-column layout - FIXED SCROLLING"""
        modules_tab = ttk.Frame(self.notebook)
        self.notebook.add(modules_tab, text="Modules")
        
        # Create main container with scrollbar
        canvas = tk.Canvas(modules_tab)
        scrollbar = ttk.Scrollbar(modules_tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Configure 4 equal columns
        for col in range(4):
            scrollable_frame.columnconfigure(col, weight=1, uniform="column")
        
        # Module controls in 4-column layout
        self.module_vars = {}
        
        # Define modules with descriptions and organize into categories
        modules_info = {
            'normalize': {
                'name': 'Audio Normalization',
                'description': 'Normalize audio levels to optimal range for consistent volume',
                'category': 'Essential'
            },
            'peak_control': {
                'name': 'Peak Control & Limiting',
                'description': 'Advanced peak detection and hard limiting before normalization',
                'category': 'Essential'
            },
            'volume_boost': {
                'name': 'Intelligent Volume Boost',
                'description': 'Smart volume boosting with LUFS targeting and dynamics preservation',
                'category': 'Essential'
            },
            'compression': {
                'name': 'Dynamic Compression',
                'description': 'Multi-threaded dynamic range compression with parallel processing',
                'category': 'Processing'
            },
            'silence_processing': {
                'name': 'Enhanced Silence Processing',
                'description': 'Audacity-style truncate silence with sentence break detection',
                'category': 'Processing'
            },
            'noise_reduction': {
                'name': 'AI Noise Reduction',
                'description': 'Advanced noise reduction with speech preservation (requires noisereduce)',
                'category': 'Processing'
            },
            'eq': {
                'name': 'Parametric EQ',
                'description': '6-band parametric equalizer with high/low pass filters',
                'category': 'Enhancement'
            },
            'limiter': {
                'name': 'Final Safety Limiter',
                'description': 'Professional brick-wall limiter with lookahead and oversampling',
                'category': 'Essential'
            }
        }
        
        # Create category sections
        categories = {'Essential': [], 'Processing': [], 'Enhancement': []}
        for module_key, info in modules_info.items():
            categories[info['category']].append((module_key, info))
        
        row = 0
        for category, modules in categories.items():
            # Category header spanning all 4 columns
            category_frame = ttk.Frame(scrollable_frame)
            category_frame.grid(row=row, column=0, columnspan=4, sticky=(tk.W, tk.E), 
                               padx=10, pady=(15, 5))
            category_frame.columnconfigure(1, weight=1)  # Make separator expand
            
            category_label = ttk.Label(category_frame, text=f"{category} Modules", 
                                     font=("Arial", 12, "bold"), foreground="navy")
            category_label.grid(row=0, column=0, sticky=tk.W)
            
            # Add separator
            separator = ttk.Separator(category_frame, orient='horizontal')
            separator.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 0))
            
            row += 1
            
            # Module cards in 4-column layout
            for i, (module_key, info) in enumerate(modules):
                col = i % 4
                module_row = row + i // 4

                # Create module card with fixed width for consistency
                card_frame = ttk.LabelFrame(scrollable_frame, text=info['name'], padding="8")
                card_frame.grid(
                    row=module_row,
                    column=col,
                    sticky=(tk.W, tk.E, tk.N),
                    padx=8, pady=6,
                    ipadx=5  # Internal padding for consistent card sizes
                )

                # Enable/disable checkbox with status indicator
                var = tk.BooleanVar(value=self.module_states.get(module_key, True))
                self.module_vars[module_key] = var

                checkbox_frame = ttk.Frame(card_frame)
                checkbox_frame.pack(fill=tk.X, pady=(0, 5))

                checkbox = ttk.Checkbutton(
                    checkbox_frame, text="Enable", variable=var,
                    style="Large.TCheckbutton"
                )
                checkbox.pack(side=tk.LEFT)

                # Status indicator
                status_label = ttk.Label(
                    checkbox_frame, text="●",
                    foreground="green" if var.get() else "red",
                    font=("Arial", 11)
                )
                status_label.pack(side=tk.RIGHT)

                # Update status indicator when checkbox changes
                def update_status(status_lbl=status_label, var=var):
                    status_lbl.config(foreground="green" if var.get() else "red")
                var.trace_add("write", lambda *args, sl=status_label, v=var: update_status(sl, v))

                # Description with consistent height
                desc_label = ttk.Label(
                    card_frame, text=info['description'],
                    wraplength=150, justify=tk.LEFT, foreground="gray60",
                    font=("Arial", 9)
                )
                desc_label.pack(fill=tk.X, pady=(0, 8))

                # Button frame
                button_frame = ttk.Frame(card_frame)
                button_frame.pack(fill=tk.X)

                # Configure button
                config_btn = ttk.Button(button_frame, text="Configure",
                              command=lambda mk=module_key: self.configure_module(mk))
                config_btn.pack(side=tk.LEFT)
                
                # Info button
                info_btn = ttk.Button(button_frame, text="?", width=3,
                              command=lambda mk=module_key: self.show_module_info(mk))
                info_btn.pack(side=tk.RIGHT)
            
            # Update row counter - add extra row for spacing between categories
            row += (len(modules) + 3) // 4 + 1
        
        # Master controls at the bottom spanning all 4 columns
        master_frame = ttk.LabelFrame(scrollable_frame, text="Master Controls", padding="10")
        master_frame.grid(row=row+1, column=0, columnspan=4, sticky=(tk.W, tk.E), 
                         padx=10, pady=20)
        
        # Split master controls into two rows for better layout
        master_row1 = ttk.Frame(master_frame)
        master_row1.pack(fill=tk.X, pady=(0, 5))
        
        master_row2 = ttk.Frame(master_frame)
        master_row2.pack(fill=tk.X)
        
        # First row - Enable/Disable/Reset
        ttk.Button(master_row1, text="Enable All", 
                  command=self.enable_all_modules).pack(side=tk.LEFT, padx=5)
        ttk.Button(master_row1, text="Disable All", 
                  command=self.disable_all_modules).pack(side=tk.LEFT, padx=5)
        ttk.Button(master_row1, text="Reset All to Default", 
                  command=self.reset_all_modules).pack(side=tk.LEFT, padx=5)
        
        # Second row - Import/Export
        ttk.Button(master_row1, text="Import Module Config", 
                  command=self.import_module_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(master_row1, text="Export Module Config", 
                  command=self.export_module_config).pack(side=tk.LEFT, padx=5)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # FIXED: Safe mouse wheel binding with error handling
        def _on_mousewheel(event):
            try:
                if canvas.winfo_exists():
                    canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            except tk.TclError:
                pass  # Canvas destroyed, ignore the event
        
        # Track this canvas for cleanup
        self.mouse_bound_widgets.append((canvas, _on_mousewheel))
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Bind canvas resize to update scroll region
        def _configure_canvas(event):
            try:
                if canvas.winfo_exists():
                    canvas.configure(scrollregion=canvas.bbox("all"))
            except tk.TclError:
                pass
        scrollable_frame.bind("<Configure>", _configure_canvas)
    
    def show_module_info(self, module_key):
        """Show detailed information about a module"""
        info_texts = {
            'normalize': """Audio Normalization

Normalizes audio levels to achieve consistent volume across your audio files.

Features:
• Peak normalization to optimal level
• Preserves original dynamic range
• Configurable headroom setting
• Works on both mono and stereo audio

Best for: Ensuring consistent loudness across multiple audio files.""",

            'peak_control': """Peak Control & Hard Limiting

Advanced peak detection and control system that prevents clipping and distortion.

Features:
• Pre-normalization peak analysis
• Smart peak detection algorithm
• Hard limiting with configurable thresholds
• Release time and lookahead controls
• Safety limiter functionality

Best for: Preventing digital clipping and maintaining audio quality.""",

            'volume_boost': """Intelligent Volume Boost

Smart volume enhancement using LUFS targeting and psychoacoustic principles.

Features:
• Adaptive gain calculation
• LUFS (Loudness Units) targeting
• Dynamic range preservation
• Automatic peak limiting
• Configurable target loudness levels

Best for: Achieving broadcast-standard loudness levels.""",

            'compression': """Dynamic Range Compression

Professional multi-threaded dynamic range compression with parallel processing.

Features:
• Configurable threshold, ratio, attack, and release
• Parallel compression mixing
• Multi-threaded processing for speed
• Knee control for smooth compression
• Automatic makeup gain

Best for: Controlling dynamic range and adding punch to audio.""",

        'silence_processing': """Enhanced Silence Processing

Advanced silence detection and truncation based on Audacity's algorithm.

Features:
• Intelligent sentence break detection
• Configurable silence thresholds
• Separate leading, trailing, and internal silence processing
• Context-aware silence preservation
• Fast numpy-based processing

⚠️ Leading Silence Warning:
By default, leading silence trimming is DISABLED to prevent cutting important audio content at the beginning of recordings. You can enable it manually if needed.

Best for: Removing unwanted silence while preserving speech flow and protecting important audio content.""",

            'noise_reduction': """AI-Powered Noise Reduction

Advanced noise reduction using spectral subtraction and machine learning.

Features:
• Speech preservation algorithms
• Configurable noise reduction strength
• Frequency and temporal masking
• Chunk-based processing for large files
• Requires noisereduce library

Best for: Cleaning up recordings with background noise.""",

            'eq': """Parametric Equalizer

Professional 6-band parametric EQ with filter options.

Features:
• 6 configurable frequency bands
• Adjustable frequency, gain, and Q factor
• High-pass and low-pass filters
• Real-time preview (when available)
• Preset save/load functionality

Best for: Tone shaping and frequency balance correction.""",

            'limiter': """Final Safety Limiter

Professional brick-wall limiter with advanced algorithms.

Features:
• Configurable threshold and release
• Lookahead processing
• Internal oversampling
• Soft knee limiting
• True peak limiting

Best for: Final stage limiting to prevent any clipping."""
        }
        
        info_text = info_texts.get(module_key, f"Information for {module_key} module.")
        
        info_window = tk.Toplevel(self.root)
        info_window.title(f"{module_key.replace('_', ' ').title()} - Information")
        info_window.geometry("500x400")
        info_window.transient(self.root)
        
        # Create scrolled text widget
        text_widget = scrolledtext.ScrolledText(info_window, wrap=tk.WORD, width=60, height=20)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text_widget.insert("1.0", info_text)
        text_widget.config(state=tk.DISABLED)
        
        # Close button
        close_button = ttk.Button(info_window, text="Close", command=info_window.destroy)
        close_button.pack(pady=10)
    
    def enable_all_modules(self):
        """Enable all modules"""
        for var in self.module_vars.values():
            var.set(True)
        self.log_message("All modules enabled")
    
    def disable_all_modules(self):
        """Disable all modules"""
        for var in self.module_vars.values():
            var.set(False)
        self.log_message("All modules disabled")
    
    def reset_all_modules(self):
        """Reset all modules to default configuration"""
        if messagebox.askyesno("Reset All Modules", "Reset all modules to default configuration?"):
            self.module_configs = copy.deepcopy(DEFAULT_MODULE_CONFIGS)
            # Reset checkboxes to default states
            defaults = self.get_default_module_states()
            for module_key, var in self.module_vars.items():
                var.set(defaults.get(module_key, True))
            self.log_message("All modules reset to default configuration")
    
    def export_module_config(self):
        """Export all module configurations to file"""
        filename = filedialog.asksaveasfilename(
            title="Export Module Configuration",
            defaultextension=".json",
            filetypes=[('JSON files', '*.json'), ('All files', '*.*')],
            initialname=f"module_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        if filename:
            try:
                # Combine module states and configurations
                export_data = {
                    'module_states': {key: var.get() for key, var in self.module_vars.items()},
                    'module_configs': self.module_configs,
                    'export_info': {
                        'version': '1.1',
                        'timestamp': datetime.now().isoformat(),
                        'description': 'Audio Processor GUI Module Configuration Export'
                    }
                }
                
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
                
                messagebox.showinfo("Export Successful", 
                                   f"Module configuration exported to:\n{os.path.basename(filename)}")
                self.log_message(f"Module configuration exported to {filename}")
                
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export configuration: {str(e)}")
    
    def import_module_config(self):
        """Import module configurations from file"""
        filename = filedialog.askopenfilename(
            title="Import Module Configuration",
            filetypes=[('JSON files', '*.json'), ('All files', '*.*')]
        )
        
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    import_data = json.load(f)
                
                # Validate import data
                if 'module_configs' not in import_data:
                    messagebox.showerror("Import Error", "Invalid configuration file format")
                    return
                
                # Import configurations
                self.module_configs.update(import_data['module_configs'])
                
                # Import module states if available
                if 'module_states' in import_data:
                    for module_key, enabled in import_data['module_states'].items():
                        if module_key in self.module_vars:
                            self.module_vars[module_key].set(enabled)
                
                messagebox.showinfo("Import Successful", 
                                   f"Module configuration imported from:\n{os.path.basename(filename)}")
                self.log_message(f"Module configuration imported from {filename}")
                
            except Exception as e:
                messagebox.showerror("Import Error", f"Failed to import configuration: {str(e)}")
    
    def configure_module(self, module_key):
        """Open enhanced module configuration window"""
        # Get current module config
        module_config = self.module_configs.get(module_key, DEFAULT_MODULE_CONFIGS.get(module_key, {}))
        
        def on_config_saved(new_config):
            self.module_configs[module_key] = new_config
            self.log_message(f"Updated {module_key} configuration")
            # FIXED: Mark configuration as modified (custom)
            self.mark_config_as_custom()
        
        # Open configuration window
        config_window = ModuleConfigWindow(
            self.root, module_key, module_config, on_config_saved
        )
    
    def mark_config_as_custom(self):
        """Mark current configuration as custom when modified"""
        # Change preset display to indicate custom modification
        current_preset = self.preset_var.get()
        if not current_preset.endswith(" (Modified)"):
            self.preset_var.set(current_preset + " (Modified)")
            self.log_message("Configuration modified - now using custom settings")
    
    def create_presets_tab(self):
        """Create presets management tab"""
        presets_tab = ttk.Frame(self.notebook)
        self.notebook.add(presets_tab, text="Presets")
        
        # Preset list
        list_frame = ttk.LabelFrame(presets_tab, text="Available Presets", padding="10")
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Treeview for presets
        columns = ('name', 'type', 'description')
        self.preset_tree = ttk.Treeview(list_frame, columns=columns, show='headings')
        
        self.preset_tree.heading('name', text='Name')
        self.preset_tree.heading('type', text='Type')
        self.preset_tree.heading('description', text='Description')
        
        self.preset_tree.column('name', width=200)
        self.preset_tree.column('type', width=100)
        self.preset_tree.column('description', width=400)
        
        # Populate preset list
        self.refresh_preset_list()
        
        # Scrollbar for preset list
        preset_scrollbar = ttk.Scrollbar(list_frame, orient="vertical", 
                                        command=self.preset_tree.yview)
        self.preset_tree.configure(yscrollcommand=preset_scrollbar.set)
        
        self.preset_tree.pack(side="left", fill="both", expand=True)
        preset_scrollbar.pack(side="right", fill="y")
        
        # Preset management buttons
        button_frame = ttk.Frame(presets_tab)
        button_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        ttk.Button(button_frame, text="Load Preset", 
                  command=self.load_selected_preset).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save as Custom", 
                  command=self.save_current_as_custom).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Delete Custom", 
                  command=self.delete_custom_preset).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Export", 
                  command=self.export_preset).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Import", 
                  command=self.import_preset).pack(side=tk.LEFT, padx=5)
    
    def create_logs_tab(self):
        """Create logs and output tab"""
        logs_tab = ttk.Frame(self.notebook)
        self.notebook.add(logs_tab, text="Logs")
        
        # Log display
        log_frame = ttk.LabelFrame(logs_tab, text="Processing Logs", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, 
                                                 width=80, height=20)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Log control buttons
        log_button_frame = ttk.Frame(logs_tab)
        log_button_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        ttk.Button(log_button_frame, text="Clear Logs", 
                  command=self.clear_logs).pack(side=tk.LEFT, padx=5)
        ttk.Button(log_button_frame, text="Copy to Clipboard", 
                  command=self.copy_logs).pack(side=tk.LEFT, padx=5)
        ttk.Button(log_button_frame, text="Save to File", 
                  command=self.save_logs).pack(side=tk.LEFT, padx=5)
        
        # Auto-scroll checkbox
        self.auto_scroll_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(log_button_frame, text="Auto-scroll", 
                       variable=self.auto_scroll_var).pack(side=tk.RIGHT, padx=5)
    
    def setup_styles(self):
        """Setup custom styles with larger fonts"""
        style = ttk.Style()
        
        # Configure larger default font
        default_font = ("Segoe UI", 11)  # Increased from default 8-9
        large_font = ("Segoe UI", 14)    # For headers
        small_font = ("Segoe UI", 10)     # For descriptions
        
        # Configure accent button style
        style.configure("Accent.TButton",
                       font=("Segoe UI", 12, "bold"))
        
        # Configure large checkbox style
        style.configure("Large.TCheckbutton",
                       font=("Segoe UI", 11, "bold"))
        
        # Configure default styles with larger fonts
        style.configure("TLabel", font=default_font)
        style.configure("TButton", font=default_font)
        style.configure("TCheckbutton", font=default_font)
        style.configure("TEntry", font=default_font)
        style.configure("TCombobox", font=default_font)
        style.configure("TLabelFrame.Label", font=("Segoe UI", 12, "bold"))
        
        # Configure Notebook tabs
        style.configure("TNotebook.Tab", font=("Segoe UI", 12))
        
        # Configure Treeview
        style.configure("Treeview", font=default_font)
        style.configure("Treeview.Heading", font=("Segoe UI", 11, "bold"))
    
    def load_settings(self):
        """Load application settings"""
        settings = self.config_manager.app_settings
        
        # Restore window geometry
        if 'window_geometry' in settings:
            self.root.geometry(settings['window_geometry'])
    
    def browse_input_file(self):
        """Browse for input audio file"""
        filetypes = [
            ('Audio files', '*.mp3 *.wav *.flac *.m4a *.aac *.ogg'),
            ('All files', '*.*')
        ]
        
        initial_dir = self.config_manager.app_settings.get('last_input_dir', str(Path.home()))
        filename = filedialog.askopenfilename(
            title="Select input audio file",
            filetypes=filetypes,
            initialdir=initial_dir
        )
        
        if filename:
            self.input_var.set(filename)
            self.config_manager.app_settings['last_input_dir'] = str(Path(filename).parent)
            
            # Auto-suggest output filename with current format
            if not self.output_var.get():
                input_path = Path(filename)
                output_ext = f".{self.format_var.get()}"
                output_path = input_path.parent / f"{input_path.stem}_processed{output_ext}"
                self.output_var.set(str(output_path))
    
    def browse_input_folder(self):
        """Browse for input folder (batch processing)"""
        initial_dir = self.config_manager.app_settings.get('last_input_dir', str(Path.home()))
        folder = filedialog.askdirectory(
            title="Select input folder for batch processing",
            initialdir=initial_dir
        )
        
        if folder:
            self.input_var.set(folder)
            self.batch_mode.set(True)
            self.config_manager.app_settings['last_input_dir'] = folder
    
    def browse_output_file(self):
        """Browse for output audio file"""
        # Get current format for file type filtering
        current_format = self.format_var.get()
        
        filetypes = []
        if current_format == "mp3":
            filetypes = [('MP3 files', '*.mp3'), ('All files', '*.*')]
            default_ext = ".mp3"
        elif current_format == "wav":
            filetypes = [('WAV files', '*.wav'), ('All files', '*.*')]
            default_ext = ".wav"
        elif current_format == "flac":
            filetypes = [('FLAC files', '*.flac'), ('All files', '*.*')]
            default_ext = ".flac"
        elif current_format == "ogg":
            filetypes = [('OGG files', '*.ogg'), ('All files', '*.*')]
            default_ext = ".ogg"
        else:
            filetypes = [
                ('MP3 files', '*.mp3'),
                ('WAV files', '*.wav'),
                ('FLAC files', '*.flac'),
                ('OGG files', '*.ogg'),
                ('All files', '*.*')
            ]
            default_ext = ".mp3"
        
        initial_dir = self.config_manager.app_settings.get('last_output_dir', str(Path.home()))
        filename = filedialog.asksaveasfilename(
            title="Select output file",
            filetypes=filetypes,
            initialdir=initial_dir,
            defaultextension=default_ext
        )
        
        if filename:
            self.output_var.set(filename)
            self.config_manager.app_settings['last_output_dir'] = str(Path(filename).parent)
            
            # Update format based on selected file extension
            file_ext = Path(filename).suffix.lower()
            if file_ext == ".mp3":
                self.format_var.set("mp3")
            elif file_ext == ".wav":
                self.format_var.set("wav")
            elif file_ext == ".flac":
                self.format_var.set("flac")
            elif file_ext == ".ogg":
                self.format_var.set("ogg")
    
    def browse_output_folder(self):
        """Browse for output folder (batch processing)"""
        initial_dir = self.config_manager.app_settings.get('last_output_dir', str(Path.home()))
        folder = filedialog.askdirectory(
            title="Select output folder for batch processing",
            initialdir=initial_dir
        )
        
        if folder:
            self.output_var.set(folder)
            self.config_manager.app_settings['last_output_dir'] = folder
    
    def on_preset_changed(self, event=None):
        """Handle preset change - FIXED VERSION"""
        preset_name = self.preset_var.get()
        
        # Remove "(Modified)" suffix if present for lookup
        clean_preset_name = preset_name.replace(" (Modified)", "")
        
        all_presets = self.config_manager.get_all_presets()
        
        if clean_preset_name in all_presets:
            # Load preset and update module states
            self.current_config = all_presets[clean_preset_name].copy()
            self.current_preset = clean_preset_name
            
            # Apply preset to modules
            self.apply_preset_to_modules(self.current_config)
            
            self.log_message(f"Loaded preset: {clean_preset_name}")
        else:
            self.log_message(f"Warning: Unknown preset '{clean_preset_name}', keeping current settings")
    
    def apply_preset_to_modules(self, preset_config):
        """Apply preset configuration to module states"""
        try:
            # Update module checkboxes based on preset
            if 'eq_enabled' in preset_config:
                if 'eq' in self.module_vars:
                    self.module_vars['eq'].set(preset_config['eq_enabled'])
            
            if 'normalize' in preset_config:
                if 'normalize' in self.module_vars:
                    self.module_vars['normalize'].set(preset_config['normalize'])
            
            # Update module configs and states
            if 'compression' in preset_config and preset_config['compression']:
                self.module_configs['compression'].update(preset_config['compression'])
                if 'compression' in self.module_vars:
                    self.module_vars['compression'].set(True)
            elif 'compression' not in preset_config or not preset_config['compression']:
                if 'compression' in self.module_vars:
                    self.module_vars['compression'].set(False)
            
            if 'noise_reduction' in preset_config and preset_config['noise_reduction']:
                self.module_configs['noise_reduction'].update(preset_config['noise_reduction'])
                if 'noise_reduction' in self.module_vars:
                    self.module_vars['noise_reduction'].set(True)
            elif 'noise_reduction' not in preset_config or not preset_config['noise_reduction']:
                if 'noise_reduction' in self.module_vars:
                    self.module_vars['noise_reduction'].set(False)
            
            if 'silence_processing' in preset_config and preset_config['silence_processing']:
                self.module_configs['silence_processing'].update(preset_config['silence_processing'])
                if 'silence_processing' in self.module_vars:
                    self.module_vars['silence_processing'].set(True)
            elif 'silence_processing' not in preset_config or not preset_config['silence_processing']:
                if 'silence_processing' in self.module_vars:
                    self.module_vars['silence_processing'].set(False)
            
            if 'limiter' in preset_config and preset_config['limiter']:
                self.module_configs['limiter'].update(preset_config['limiter'])
                if 'limiter' in self.module_vars:
                    self.module_vars['limiter'].set(True)
            elif 'limiter' not in preset_config or not preset_config['limiter']:
                if 'limiter' in self.module_vars:
                    self.module_vars['limiter'].set(False)
            
            if 'peak_control' in preset_config and preset_config['peak_control']:
                self.module_configs['peak_control'].update(preset_config['peak_control'])
                if 'peak_control' in self.module_vars:
                    self.module_vars['peak_control'].set(preset_config['peak_control'].get('enabled', True))
            
            if 'volume_boost' in preset_config and preset_config['volume_boost']:
                self.module_configs['volume_boost'].update(preset_config['volume_boost'])
                if 'volume_boost' in self.module_vars:
                    self.module_vars['volume_boost'].set(preset_config['volume_boost'].get('enabled', True))
            
            # Update gain
            if 'gain_db' in preset_config:
                self.gain_var.set(str(preset_config['gain_db']))
                
        except Exception as e:
            self.log_message(f"Warning: Error applying preset to modules: {e}")
    
    def load_custom_preset(self):
        """Load custom preset from file"""
        filetypes = [('JSON files', '*.json'), ('All files', '*.*')]
        filename = filedialog.askopenfilename(
            title="Load custom preset",
            filetypes=filetypes
        )
        
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    preset_data = json.load(f)
                
                self.current_config = preset_data
                self.apply_preset_to_modules(preset_data)
                self.preset_var.set("Custom (Loaded)")
                self.log_message(f"Loaded custom preset from {filename}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load preset: {str(e)}")
    
    def save_custom_preset(self):
        """Save current configuration as custom preset"""
        name = simpledialog.askstring("Save Preset", "Enter preset name:")
        if name:
            # Update current config with module states
            self.update_config_from_gui()
            
            # Add to custom presets
            self.config_manager.custom_presets[name] = {
                'name': name,
                'description': f'Custom preset created {datetime.now().strftime("%Y-%m-%d %H:%M")}',
                **self.current_config
            }
            
            self.config_manager.save_custom_presets()
            self.refresh_preset_list()
            self.preset_var.set(name)
            self.log_message(f"Saved custom preset: {name}")
    
    def refresh_preset_list(self):
        """Refresh the preset list in presets tab"""
        # Clear existing items
        for item in self.preset_tree.get_children():
            self.preset_tree.delete(item)
        
        # Add default presets
        for name, config in self.config_manager.default_presets.items():
            self.preset_tree.insert('', 'end', values=(
                name, 'Default', config.get('description', '')
            ))
        
        # Add custom presets
        for name, config in self.config_manager.custom_presets.items():
            self.preset_tree.insert('', 'end', values=(
                name, 'Custom', config.get('description', '')
            ))
    
    def load_selected_preset(self):
        """Load the selected preset from the list"""
        selection = self.preset_tree.selection()
        if selection:
            item = self.preset_tree.item(selection[0])
            preset_name = item['values'][0]
            
            all_presets = self.config_manager.get_all_presets()
            if preset_name in all_presets:
                self.current_config = all_presets[preset_name].copy()
                self.apply_preset_to_modules(self.current_config)
                self.preset_var.set(preset_name)
                self.log_message(f"Loaded preset: {preset_name}")
    
    def save_current_as_custom(self):
        """Save current settings as custom preset"""
        self.save_custom_preset()
    
    def delete_custom_preset(self):
        """Delete selected custom preset"""
        selection = self.preset_tree.selection()
        if selection:
            item = self.preset_tree.item(selection[0])
            preset_name = item['values'][0]
            preset_type = item['values'][1]
            
            if preset_type == 'Custom':
                if messagebox.askyesno("Confirm Delete", 
                                     f"Delete custom preset '{preset_name}'?"):
                    del self.config_manager.custom_presets[preset_name]
                    self.config_manager.save_custom_presets()
                    self.refresh_preset_list()
                    self.log_message(f"Deleted custom preset: {preset_name}")
            else:
                messagebox.showinfo("Info", "Cannot delete default presets")
    
    def export_preset(self):
        """Export preset to file"""
        selection = self.preset_tree.selection()
        if selection:
            item = self.preset_tree.item(selection[0])
            preset_name = item['values'][0]
            
            all_presets = self.config_manager.get_all_presets()
            if preset_name in all_presets:
                filename = filedialog.asksaveasfilename(
                    title="Export preset",
                    defaultextension=".json",
                    filetypes=[('JSON files', '*.json'), ('All files', '*.*')],
                    initialname=f"{preset_name}.json"
                )
                
                if filename:
                    try:
                        with open(filename, 'w', encoding='utf-8') as f:
                            json.dump(all_presets[preset_name], f, indent=2, ensure_ascii=False)
                        self.log_message(f"Exported preset to {filename}")
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to export: {str(e)}")
    
    def import_preset(self):
        """Import preset from file"""
        filename = filedialog.askopenfilename(
            title="Import preset",
            filetypes=[('JSON files', '*.json'), ('All files', '*.*')]
        )
        
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    preset_data = json.load(f)
                
                name = simpledialog.askstring(
                    "Import Preset", 
                    "Enter name for imported preset:",
                    initialvalue=Path(filename).stem
                )
                
                if name:
                    self.config_manager.custom_presets[name] = {
                        'name': name,
                        'description': f'Imported from {Path(filename).name}',
                        **preset_data
                    }
                    
                    self.config_manager.save_custom_presets()
                    self.refresh_preset_list()
                    self.log_message(f"Imported preset: {name}")
                    
            except Exception as e:
                messagebox.showerror("Error", f"Failed to import: {str(e)}")
    
    def update_config_from_gui(self):
        """Update current configuration from GUI settings and module states - FIXED"""
    def update_config_from_gui(self):
        """Update current configuration from GUI settings and module states - FIXED"""
        # Start with current base preset if valid
        current_preset_name = self.preset_var.get().replace(" (Modified)", "")
        if current_preset_name in self.config_manager.get_all_presets():
            self.current_config = self.config_manager.get_all_presets()[current_preset_name].copy()
        
        # Update gain
        try:
            gain_value = float(self.gain_var.get())
            self.current_config['gain_db'] = gain_value
        except ValueError:
            pass
        
        # Update configuration based on module states
        for module_key, config in self.module_configs.items():
            if module_key in self.module_vars:
                enabled = self.module_vars[module_key].get()
                
                # Map to script configuration format
                if module_key == 'normalize':
                    self.current_config['normalize'] = enabled
                elif module_key == 'eq':
                    self.current_config['eq_enabled'] = enabled
                    if enabled:
                        self.current_config['eq_bands'] = config.get('bands', {})
                elif module_key == 'peak_control':
                    if enabled:
                        self.current_config['peak_control'] = config
                    else:
                        self.current_config['peak_control'] = {'enabled': False}
                elif module_key == 'volume_boost':
                    if enabled:
                        self.current_config['volume_boost'] = config
                    else:
                        self.current_config['volume_boost'] = {'enabled': False}
                elif module_key == 'compression':
                    if enabled:
                        self.current_config['compression'] = config
                    else:
                        self.current_config['compression'] = None
                elif module_key == 'noise_reduction':
                    if enabled:
                        self.current_config['noise_reduction'] = config
                    else:
                        self.current_config['noise_reduction'] = None
                elif module_key == 'silence_processing':
                    if enabled:
                        silence_config = config.copy()
                        
                        # ✅ CRITICAL: ต้องส่งค่า process_leading และ process_trailing ไปด้วย
                        silence_config['enabled'] = True
                        silence_config['process_leading'] = silence_config.get('process_leading', True)
                        silence_config['process_trailing'] = silence_config.get('process_trailing', True)
                        
                        """
                        # ✅ เพิ่ม debug log
                        print(f"DEBUG GUI - Silence config being sent:")
                        print(f"  process_leading: {silence_config.get('process_leading')}")
                        print(f"  process_trailing: {silence_config.get('process_trailing')}")
                        """
                        
                        self.current_config['silence_processing'] = silence_config
                    else:
                        self.current_config['silence_processing'] = None
                elif module_key == 'limiter':
                    if enabled:
                        self.current_config['limiter'] = config
                    else:
                        self.current_config['limiter'] = None

    def start_processing(self):
        """Start audio processing"""
        if self.processing:
            return
        
        # Validate inputs
        input_path = self.input_var.get().strip()
        output_path = self.output_var.get().strip()
        
        if not input_path:
            messagebox.showerror("Error", "Please select an input file or folder")
            return
        
        if not output_path:
            messagebox.showerror("Error", "Please select an output file or folder")
            return
        
        # Update config from GUI
        self.update_config_from_gui()
        
        # Disable controls
        self.processing = True
        self.process_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_var.set("Processing...")
        
        # Start processing thread
        self.process_thread = threading.Thread(
            target=self.process_audio_thread,
            args=(input_path, output_path),
            daemon=True
        )
        self.process_thread.start()
    
    def process_audio_thread(self, input_path, output_path):
        """Audio processing thread with PATH FIXES"""
        try:
            # FIXED: Normalize paths to avoid forward/backward slash issues
            input_path = os.path.normpath(input_path)
            output_path = os.path.normpath(output_path)
            
            self.process_queue.put(("log", f"Normalized input: {input_path}"))
            self.process_queue.put(("log", f"Normalized output: {output_path}"))
            
            # Check if input file exists
            if not os.path.exists(input_path):
                self.process_queue.put(("error", f"Input file not found: {input_path}"))
                return
                
            # Check if we can write to output directory
            output_dir = os.path.dirname(output_path)
            if not os.path.exists(output_dir):
                try:
                    os.makedirs(output_dir, exist_ok=True)
                    self.process_queue.put(("log", f"Created output directory: {output_dir}"))
                except Exception as e:
                    self.process_queue.put(("error", f"Cannot create output directory: {e}"))
                    return
            
            # Create temporary config file
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json', encoding='utf-8') as f:
                json.dump(self.current_config, f, indent=2, ensure_ascii=False)
                config_file = f.name
                
            self.process_queue.put(("log", f"Created config file: {config_file}"))

            # Build command with proper path handling
            script_path = "audio_processor_pro-v5.8.py"
            
            # FIXED: Use absolute paths and proper quoting
            if not os.path.exists(script_path):
                script_path = os.path.join(os.path.dirname(__file__), script_path)
                if not os.path.exists(script_path):
                    self.process_queue.put(("error", f"Cannot find script: {script_path}"))
                    return
            
            # Use absolute paths for all file arguments
            abs_script = os.path.abspath(script_path)
            abs_input = os.path.abspath(input_path)
            abs_output = os.path.abspath(output_path)
            abs_config = os.path.abspath(config_file)
            
            cmd = [sys.executable, abs_script, abs_input]
            
            if not self.batch_mode.get():
                cmd.append(abs_output)
            
            # Configuration handling
            current_preset_name = self.preset_var.get().replace(" (Modified)", "")
            available_presets = ['sermon_ultra', 'broadcast_pro', 'podcast_pro', 'music_mastering']
            
            is_heavily_modified = (
                current_preset_name not in available_presets or
                self.preset_var.get().endswith(" (Modified)") or
                current_preset_name.startswith("Custom")
            )
            
            if is_heavily_modified:
                self.process_queue.put(("log", "Using custom configuration"))
                cmd.extend(["--load-config", abs_config])
            else:
                self.process_queue.put(("log", f"Using preset: {current_preset_name}"))
                cmd.extend(["--preset", current_preset_name, "--load-config", abs_config])
            
            # Add other arguments
            cmd.extend(["--speed-preset", self.speed_preset_var.get()])
            
            if self.batch_mode.get():
                cmd.append("--batch")
            if self.overwrite_var.get():
                cmd.append("--overwrite")
            if self.debug_var.get():
                cmd.append("--debug-speed")
            
            # Thread setting
            threads = self.threads_var.get().strip()
            if threads and threads != "auto":
                try:
                    int(threads)
                    cmd.extend(["--threads", threads])
                except ValueError:
                    pass
            
            # Gain setting
            try:
                gain_value = float(self.gain_var.get())
                if abs(gain_value) > 0.1:
                    cmd.extend(["--gain", str(gain_value)])
            except ValueError:
                pass
            
            # Format options
            format_val = self.format_var.get()
            if format_val and format_val != "mp3":
                cmd.extend(["--format", format_val])
            
            bitrate_val = self.bitrate_var.get()
            if bitrate_val and bitrate_val != "auto" and bitrate_val != "192k":
                cmd.extend(["--bitrate", bitrate_val])
            
            sample_rate_val = self.sample_rate_var.get()
            if sample_rate_val and sample_rate_val != "auto":
                cmd.extend(["--sample-rate", sample_rate_val])
            
            # Module overrides
            if not self.module_vars.get('eq', tk.BooleanVar()).get():
                cmd.append("--disable-eq")
            if not self.module_vars.get('compression', tk.BooleanVar()).get():
                cmd.append("--no-compression")
            if not self.module_vars.get('noise_reduction', tk.BooleanVar()).get():
                cmd.append("--no-noise-reduction")
            if not self.module_vars.get('limiter', tk.BooleanVar()).get():
                cmd.append("--no-limiter")
            if not self.module_vars.get('silence_processing', tk.BooleanVar()).get():
                cmd.append("--no-silence-processing")
            if not self.module_vars.get('normalize', tk.BooleanVar()).get():
                cmd.append("--no-normalize")

            # Silence processing options
            silence_config = self.module_configs.get('silence_processing', {})
            if silence_config:
                process_leading = silence_config.get('process_leading', False)
                process_trailing = silence_config.get('process_trailing', True)
                
                if not process_leading:
                    cmd.append("--no-trim-leading")
                    self.process_queue.put(("log", "Leading silence trimming disabled"))
                else:
                    cmd.append("--trim-leading")
                    self.process_queue.put(("log", "Leading silence trimming enabled"))
                    
                if not process_trailing:
                    cmd.append("--no-trim-trailing")  
                    self.process_queue.put(("log", "Trailing silence trimming disabled"))
                else:
                    cmd.append("--trim-trailing")
                    self.process_queue.put(("log", "Trailing silence trimming enabled"))
            
            # IMPROVED: Better process execution with timeout
            self.process_queue.put(("status", "Starting audio processing..."))
            self.process_queue.put(("log", f"Command: {' '.join(cmd)}"))
            
            # Set working directory to script directory
            working_dir = os.path.dirname(abs_script)
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                encoding='utf-8',
                bufsize=1,
                cwd=working_dir,  # Set working directory
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            
            self.process_queue.put(("log", f"Process started with PID: {process.pid}"))
            
            # Read output with timeout handling
            import time
            last_output_time = time.time()
            timeout_seconds = 600  # 10 minute timeout for large files

            while True:
                # Check if process finished
                if process.poll() is not None:
                    # Process finished, read remaining output
                    remaining = process.stdout.read()
                    if remaining:
                        for line in remaining.split('\n'):
                            if line.strip():
                                self.process_queue.put(("log", line.strip()))
                    break
                
                # Try to read a line with timeout - FIXED for Windows
                try:
                    # Use simple polling for Windows compatibility
                    line = process.stdout.readline()
                    if line:
                        line = line.strip()
                        if line:
                            self.process_queue.put(("log", line))
                            last_output_time = time.time()
                            
                            # Simple progress estimation based on log messages
                            line_lower = line.lower()
                            if "loading audio" in line_lower:
                                self.process_queue.put(("progress", 10))
                            elif "channel processing" in line_lower:
                                self.process_queue.put(("progress", 20))
                            elif "peak detection" in line_lower:
                                self.process_queue.put(("progress", 30))
                            elif "normalization" in line_lower:
                                self.process_queue.put(("progress", 40))
                            elif "compression" in line_lower:
                                self.process_queue.put(("progress", 50))
                            elif "silence processing" in line_lower:
                                self.process_queue.put(("progress", 70))
                            elif "exporting" in line_lower:
                                self.process_queue.put(("progress", 90))
                    else:
                        # No output available, wait a bit
                        time.sleep(0.1)
                            
                except Exception as e:
                    self.process_queue.put(("log", f"Output read error: {e}"))
                    time.sleep(1)
                
                # Check timeout
                if time.time() - last_output_time > timeout_seconds:
                    self.process_queue.put(("error", f"Process timeout after {timeout_seconds} seconds"))
                    try:
                        process.terminate()
                        time.sleep(5)
                        if process.poll() is None:
                            process.kill()
                            self.process_queue.put(("log", "Process forcefully terminated"))
                    except:
                        pass
                    break
            
            # Get final return code
            return_code = process.poll()
            
            if return_code == 0:
                self.process_queue.put(("status", "Processing completed successfully!"))
                self.process_queue.put(("progress", 100))
            else:
                self.process_queue.put(("status", "Processing failed"))
                self.process_queue.put(("error", f"Process exited with code {return_code}"))
            
        except Exception as e:
            self.process_queue.put(("error", f"Processing error: {str(e)}"))
            import traceback
            self.process_queue.put(("log", f"Full traceback: {traceback.format_exc()}"))
        
        finally:
            # Cleanup
            if 'config_file' in locals():
                try:
                    os.unlink(config_file)
                    self.process_queue.put(("log", "Cleaned up temporary config file"))
                except:
                    pass
            
            self.process_queue.put(("done", None))
        
    def stop_processing(self):
        """Stop audio processing"""
        if self.process_thread and self.process_thread.is_alive():
            self.log_message("Stop requested - please wait for current operation to complete")
        
        self.processing = False
        self.process_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_var.set("Stopped")
    
    def check_queue(self):
        """Check for messages from processing thread"""
        try:
            while True:
                try:
                    msg_type, data = self.process_queue.get_nowait()
                    
                    if msg_type == "log":
                        self.log_message(data)
                    elif msg_type == "status":
                        self.status_var.set(data)
                    elif msg_type == "progress":
                        self.progress_var.set(data)
                    elif msg_type == "error":
                        self.log_message(f"ERROR: {data}")
                        messagebox.showerror("Processing Error", data)
                    elif msg_type == "done":
                        self.processing = False
                        self.process_button.config(state=tk.NORMAL)
                        self.stop_button.config(state=tk.DISABLED)
                        break
                        
                except queue.Empty:
                    break
                    
        except Exception as e:
            print(f"Queue check error: {e}")
        
        # Schedule next check
        self.root.after(100, self.check_queue)
    
    def clear_fields(self):
        """Clear input/output fields"""
        self.input_var.set("")
        self.output_var.set("")
        self.batch_mode.set(False)
        self.overwrite_var.set(False)
        
        # Reset format options to defaults
        self.format_var.set("mp3")
        self.bitrate_var.set("192k") 
        self.sample_rate_var.set("auto")
    
    def log_message(self, message):
        """Add message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"
        
        self.log_text.insert(tk.END, formatted_message)
        
        if self.auto_scroll_var.get():
            self.log_text.see(tk.END)
    
    def clear_logs(self):
        """Clear log display"""
        self.log_text.delete(1.0, tk.END)
    
    def copy_logs(self):
        """Copy logs to clipboard"""
        logs = self.log_text.get(1.0, tk.END)
        self.root.clipboard_clear()
        self.root.clipboard_append(logs)
        messagebox.showinfo("Info", "Logs copied to clipboard")
    
    def save_logs(self):
        """Save logs to file"""
        filename = filedialog.asksaveasfilename(
            title="Save logs",
            defaultextension=".txt",
            filetypes=[('Text files', '*.txt'), ('All files', '*.*')],
            initialname=f"audio_processor_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        
        if filename:
            try:
                logs = self.log_text.get(1.0, tk.END)
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(logs)
                self.log_message(f"Logs saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save logs: {str(e)}")
    
    def cleanup_mouse_bindings(self):
        """Clean up mouse wheel bindings to prevent errors"""
        for widget, callback in self.mouse_bound_widgets:
            try:
                if widget.winfo_exists():
                    widget.unbind_all("<MouseWheel>")
            except tk.TclError:
                pass  # Widget already destroyed
        self.mouse_bound_widgets.clear()
    
    def on_closing(self):
        """Handle application closing"""
        # Clean up mouse wheel bindings
        self.cleanup_mouse_bindings()
        
        # Save settings
        self.config_manager.app_settings['window_geometry'] = self.root.geometry()
        self.config_manager.save_app_settings()
        
        # Stop any running processes
        if self.processing:
            if messagebox.askyesno("Confirm Exit", 
                                 "Processing is running. Exit anyway?"):
                self.root.destroy()
        else:
            self.root.destroy()
    
    def run(self):
        """Run the application"""
        # Set up close handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Center window
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"+{x}+{y}")
        
        # Start application
        self.log_message("Enhanced Audio Processor GUI v1.1 - FIXED VERSION started")
        self.log_message("Fixed: Custom preset handling, mouse wheel scrolling, configuration sync")
        self.root.mainloop()


def main():
    """Main entry point"""
    # Check if original script exists
    script_path = "audio_processor_pro-v5.8.py"
    if not os.path.exists(script_path):
        messagebox.showerror(
            "Script Not Found",
            f"Cannot find {script_path}\n\n"
            "Please ensure the audio processor script is in the same directory as this GUI."
        )
        return
    
    # Create and run GUI
    app = AudioProcessorGUI()
    app.run()


if __name__ == "__main__":
    main()