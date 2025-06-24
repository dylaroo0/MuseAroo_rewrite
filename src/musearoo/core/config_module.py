#!/usr/bin/env python3
"""
Music MuseAroo Configuration System
==================================

Centralized configuration management for Music MuseAroo v2.0.0
Handles environment settings, paths, and feature flags.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PathConfig:
    """Directory and file path configurations."""
    # Core directories
    data_dir: str = "data"
    output_dir: str = "reports"
    plugin_dir: str = "src/plugins"
    log_dir: str = "logs"
    temp_dir: str = "temp"
    models_dir: str = "models"
    
    # Data subdirectories
    audio_dir: str = "data/audio"
    midi_dir: str = "data/midi"
    musicxml_dir: str = "data/musicxml"
    
    # Log files
    main_log: str = "logs/musearoo.log"
    error_log: str = "logs/errors.log"
    plugin_log: str = "logs/plugins.log"
    
    def __post_init__(self):
        """Convert relative paths to Path objects."""
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if isinstance(value, str):
                setattr(self, field_name, Path(value))

@dataclass
class AudioConfig:
    """Audio processing configuration."""
    sample_rate: int = 22050
    hop_length: int = 512
    n_fft: int = 2048
    n_mels: int = 128
    max_duration: float = 300.0  # 5 minutes max
    supported_formats: List[str] = field(default_factory=lambda: [
        '.wav', '.mp3', '.flac', '.m4a', '.aiff', '.ogg'
    ])
    
    # Audio analysis settings
    tempo_min: int = 60
    tempo_max: int = 200
    onset_threshold: float = 0.5
    beat_threshold: float = 0.3

@dataclass
class MidiConfig:
    """MIDI processing configuration."""
    default_velocity: int = 80
    default_tempo: float = 120.0
    ticks_per_beat: int = 480
    max_duration: float = 600.0  # 10 minutes max
    supported_formats: List[str] = field(default_factory=lambda: [
        '.mid', '.midi'
    ])
    
    # MIDI analysis settings
    min_note_duration: float = 0.1
    velocity_threshold: int = 10
    chord_detection_window: float = 0.5

@dataclass
class PluginConfig:
    """Plugin system configuration."""
    auto_discover: bool = True
    parallel_execution: bool = False
    timeout_seconds: int = 300
    max_memory_mb: int = 1024
    
    # Plugin directories to search
    search_paths: List[str] = field(default_factory=lambda: [
        "src/plugins",
        "plugins",
        "~/.music_musearoo/plugins"
    ])
    
    # Phase execution settings
    phase_timeout: Dict[int, int] = field(default_factory=lambda: {
        1: 60,   # Core analysis - 1 minute
        2: 180,  # Advanced analysis - 3 minutes
        3: 300,  # Generation - 5 minutes
        4: 120,  # Post-processing - 2 minutes
        5: 60    # Export - 1 minute
    })

@dataclass
class APIConfig:
    """API server configuration."""
    host: str = "127.0.0.1"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    log_level: str = "info"
    
    # File upload settings
    max_file_size_mb: int = 100
    upload_timeout: int = 300
    temp_file_cleanup: bool = True
    temp_file_ttl: int = 3600  # 1 hour
    
    # CORS settings
    allow_origins: List[str] = field(default_factory=lambda: ["*"])
    allow_methods: List[str] = field(default_factory=lambda: ["*"])
    allow_headers: List[str] = field(default_factory=lambda: ["*"])

@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    
    # File logging
    file_logging: bool = True
    file_level: str = "DEBUG"
    max_file_size_mb: int = 10
    backup_count: int = 5
    
    # Console logging
    console_logging: bool = True
    console_level: str = "INFO"
    
    # Plugin logging
    plugin_logging: bool = True
    plugin_log_file: str = "logs/plugins.log"

@dataclass
class FeatureFlags:
    """Feature flags for experimental functionality."""
    # Generation features
    enable_drummaroo: bool = True
    enable_bass_generation: bool = True
    enable_melody_variation: bool = True
    enable_chord_generation: bool = False  # Experimental
    
    # Analysis features
    enable_audio_analysis: bool = True
    enable_genre_detection: bool = True
    enable_mood_analysis: bool = False  # Experimental
    enable_structure_analysis: bool = True
    
    # API features
    enable_batch_processing: bool = True
    enable_streaming_upload: bool = False  # Experimental
    enable_webhook_callbacks: bool = False  # Experimental
    
    # Performance features
    enable_caching: bool = True
    enable_parallel_plugins: bool = False  # Experimental
    enable_gpu_acceleration: bool = False  # Experimental
    
    # JSYMBOLIC feature flag
    ENABLE_JSYMBOLIC: bool = True

@dataclass
class MuseArooConfig:
    """Main configuration container."""
    # Core configuration sections
    paths: PathConfig = field(default_factory=PathConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    midi: MidiConfig = field(default_factory=MidiConfig)
    plugins: PluginConfig = field(default_factory=PluginConfig)
    api: APIConfig = field(default_factory=APIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    features: FeatureFlags = field(default_factory=FeatureFlags)
    
    # Metadata
    version: str = "2.0.0"
    environment: str = "development"
    debug: bool = False

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION LOADING AND MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_file: Optional[str] = None, 
               environment: Optional[str] = None) -> MuseArooConfig:
    """
    Load configuration from file and environment variables.
    
    Args:
        config_file: Path to configuration file (JSON)
        environment: Environment name (development, staging, production)
        
    Returns:
        Loaded configuration object
    """
    # Start with default configuration
    config = MuseArooConfig()
    
    # Override environment if specified
    if environment:
        config.environment = environment
        config.debug = environment == "development"
    
    # Load from environment variables
    _load_from_environment(config)
    
    # Load from configuration file
    if config_file:
        _load_from_file(config, config_file)
    else:
        # Try to find default config files
        for default_file in ["config.json", "musearoo.json", "config/default.json"]:
            if Path(default_file).exists():
                _load_from_file(config, default_file)
                break
    
    # Apply environment-specific overrides
    _apply_environment_overrides(config)
    
    # Validate configuration
    _validate_config(config)
    
    return config

def _load_from_environment(config: MuseArooConfig):
    """Load configuration from environment variables."""
    
    # Paths
    if "MUSEAROO_DATA_DIR" in os.environ:
        config.paths.data_dir = Path(os.environ["MUSEAROO_DATA_DIR"])
    if "MUSEAROO_OUTPUT_DIR" in os.environ:
        config.paths.output_dir = Path(os.environ["MUSEAROO_OUTPUT_DIR"])
    if "MUSEAROO_PLUGIN_DIR" in os.environ:
        config.paths.plugin_dir = Path(os.environ["MUSEAROO_PLUGIN_DIR"])
    
    # API
    if "MUSEAROO_API_HOST" in os.environ:
        config.api.host = os.environ["MUSEAROO_API_HOST"]
    if "MUSEAROO_API_PORT" in os.environ:
        config.api.port = int(os.environ["MUSEAROO_API_PORT"])
    if "MUSEAROO_API_WORKERS" in os.environ:
        config.api.workers = int(os.environ["MUSEAROO_API_WORKERS"])
    
    # Logging
    if "MUSEAROO_LOG_LEVEL" in os.environ:
        config.logging.level = os.environ["MUSEAROO_LOG_LEVEL"].upper()
    
    # Environment
    if "MUSEAROO_ENV" in os.environ:
        config.environment = os.environ["MUSEAROO_ENV"]
    if "MUSEAROO_DEBUG" in os.environ:
        config.debug = os.environ["MUSEAROO_DEBUG"].lower() in ("true", "1", "yes")

def _load_from_file(config: MuseArooConfig, config_file: str):
    """Load configuration from JSON file."""
    try:
        with open(config_file, 'r') as f:
            file_config = json.load(f)
        
        # Update configuration with file values
        _update_config_from_dict(config, file_config)
        logger.info(f"Loaded configuration from: {config_file}")
        
    except FileNotFoundError:
        logger.warning(f"Configuration file not found: {config_file}")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in configuration file {config_file}: {e}")
    except Exception as e:
        logger.error(f"Error loading configuration file {config_file}: {e}")

def _update_config_from_dict(config: MuseArooConfig, config_dict: Dict[str, Any]):
    """Update configuration object from dictionary."""
    for section_name, section_data in config_dict.items():
        if hasattr(config, section_name) and isinstance(section_data, dict):
            section_obj = getattr(config, section_name)
            
            for key, value in section_data.items():
                if hasattr(section_obj, key):
                    # Handle Path objects
                    if isinstance(getattr(section_obj, key), Path):
                        setattr(section_obj, key, Path(value))
                    else:
                        setattr(section_obj, key, value)

def _apply_environment_overrides(config: MuseArooConfig):
    """Apply environment-specific configuration overrides."""
    if config.environment == "production":
        # Production overrides
        config.debug = False
        config.logging.level = "WARNING"
        config.logging.console_level = "WARNING"
        config.api.reload = False
        config.features.enable_parallel_plugins = True
        
    elif config.environment == "staging":
        # Staging overrides
        config.debug = False
        config.logging.level = "INFO"
        config.api.reload = False
        
    elif config.environment == "development":
        # Development overrides
        config.debug = True
        config.logging.level = "DEBUG"
        config.logging.console_level = "DEBUG"
        config.api.reload = True

def _validate_config(config: MuseArooConfig):
    """Validate configuration values."""
    # Validate port range
    if not (1 <= config.api.port <= 65535):
        raise ValueError(f"Invalid API port: {config.api.port}")
    
    # Validate worker count
    if config.api.workers < 1:
        raise ValueError(f"Invalid worker count: {config.api.workers}")
    
    # Validate file size limits
    if config.api.max_file_size_mb < 1:
        raise ValueError(f"Invalid max file size: {config.api.max_file_size_mb}")
    
    # Validate timeout values
    if config.plugins.timeout_seconds < 10:
        raise ValueError(f"Plugin timeout too low: {config.plugins.timeout_seconds}")

def ensure_directories(config: MuseArooConfig):
    """Ensure all required directories exist."""
    directories = [
        config.paths.data_dir,
        config.paths.output_dir,
        config.paths.log_dir,
        config.paths.temp_dir,
        config.paths.models_dir,
        config.paths.audio_dir,
        config.paths.midi_dir,
        config.paths.musicxml_dir,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {directory}")

def setup_logging(config: MuseArooConfig):
    """Setup logging based on configuration."""
    import logging.handlers
    
    # Clear existing handlers
    logging.getLogger().handlers.clear()
    
    # Setup formatting
    formatter = logging.Formatter(
        config.logging.format,
        datefmt=config.logging.date_format
    )
    
    # Console handler
    if config.logging.console_logging:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, config.logging.console_level))
        console_handler.setFormatter(formatter)
        logging.getLogger().addHandler(console_handler)
    
    # File handler
    if config.logging.file_logging:
        config.paths.log_dir.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            config.paths.main_log,
            maxBytes=config.logging.max_file_size_mb * 1024 * 1024,
            backupCount=config.logging.backup_count
        )
        file_handler.setLevel(getattr(logging, config.logging.file_level))
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)
    
    # Set root logger level
    logging.getLogger().setLevel(getattr(logging, config.logging.level))

def save_config(config: MuseArooConfig, config_file: str):
    """Save configuration to JSON file."""
    # Convert config to dictionary
    config_dict = _config_to_dict(config)
    
    # Save to file
    with open(config_file, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)
    
    logger.info(f"Configuration saved to: {config_file}")

def _config_to_dict(config: MuseArooConfig) -> Dict[str, Any]:
    """Convert configuration object to dictionary."""
    result = {}
    
    for field_name in config.__dataclass_fields__:
        field_value = getattr(config, field_name)
        
        if hasattr(field_value, '__dataclass_fields__'):
            # Nested dataclass
            result[field_name] = _dataclass_to_dict(field_value)
        else:
            result[field_name] = field_value
    
    return result

def _dataclass_to_dict(obj: Any) -> Dict[str, Any]:
    """Convert dataclass to dictionary."""
    result = {}
    
    for field_name in obj.__dataclass_fields__:
        field_value = getattr(obj, field_name)
        
        if isinstance(field_value, Path):
            result[field_name] = str(field_value)
        elif isinstance(field_value, list):
            result[field_name] = field_value
        elif isinstance(field_value, dict):
            result[field_name] = field_value
        else:
            result[field_name] = field_value
    
    return result

def print_config_summary(config: MuseArooConfig):
    """Print a summary of the current configuration."""
    print("🔧 Music MuseAroo Configuration")
    print("=" * 50)
    print(f"Version: {config.version}")
    print(f"Environment: {config.environment}")
    print(f"Debug Mode: {config.debug}")
    print()
    
    print("📁 Paths:")
    print(f"  Data: {config.paths.data_dir}")
    print(f"  Output: {config.paths.output_dir}")
    print(f"  Plugins: {config.paths.plugin_dir}")
    print(f"  Logs: {config.paths.log_dir}")
    print()
    
    print("🎵 Audio Settings:")
    print(f"  Sample Rate: {config.audio.sample_rate}")
    print(f"  Supported Formats: {config.audio.supported_formats}")
    print()
    
    print("🎹 MIDI Settings:")
    print(f"  Default Tempo: {config.midi.default_tempo}")
    print(f"  Supported Formats: {config.midi.supported_formats}")
    print()
    
    print("🌐 API Settings:")
    print(f"  Host: {config.api.host}")
    print(f"  Port: {config.api.port}")
    print(f"  Workers: {config.api.workers}")
    print()
    
    print("🔌 Plugin Settings:")
    print(f"  Auto Discovery: {config.plugins.auto_discover}")
    print(f"  Timeout: {config.plugins.timeout_seconds}s")
    print(f"  Search Paths: {config.plugins.search_paths}")
    print()

def validate_environment() -> bool:
    """Validate the environment and dependencies."""
    print("🔍 Validating Environment...")
    
    valid = True
    
    # Check Python version
    import sys
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        valid = False
    else:
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")
    
    # Check core dependencies
    dependencies = [
        'librosa', 'pretty_midi', 'music21', 'fastapi', 
        'numpy', 'matplotlib', 'pathlib'
    ]
    
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep}")
        except ImportError:
            print(f"❌ {dep} - missing")
            valid = False
    
    # Check directories
    config = get_config()
    directories = [
        config.paths.data_dir,
        config.paths.output_dir,
        config.paths.log_dir,
        config.paths.temp_dir
    ]
    
    for directory in directories:
        if directory.exists():
            print(f"✅ {directory}")
        else:
            print(f"⚠️  {directory} - will be created")
    
    return valid

# Global configuration instance
_global_config: Optional[MuseArooConfig] = None

def get_config() -> MuseArooConfig:
    """Get the global configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = load_config()
    return _global_config

def set_config(config: MuseArooConfig):
    """Set the global configuration instance."""
    global _global_config
    _global_config = config

if __name__ == "__main__":
    # Demo/test configuration loading
    config = load_config()
    print_config_summary(config)
    print(f"\nValidation: {'✅ PASS' if validate_environment() else '❌ FAIL'}")