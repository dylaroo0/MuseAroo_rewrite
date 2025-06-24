#!/usr/bin/env python3
"""
MuseAroo Configuration Manager v2.1
Centralized configuration system for the entire MuseAroo ecosystem
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field, asdict
from enum import Enum


class Environment(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class PathConfig:
    """File path configuration"""
    data_dir: str = "data"
    output_dir: str = "reports"
    temp_dir: str = "temp"
    models_dir: str = "models"
    logs_dir: str = "logs"
    cache_dir: str = "cache"
    plugins_dir: str = "plugins"
    
    def ensure_directories(self):
        """Create all directories if they don't exist"""
        for field_name, dir_path in asdict(self).items():
            Path(dir_path).mkdir(parents=True, exist_ok=True)


@dataclass
class AudioConfig:
    """Audio processing configuration"""
    sample_rate: int = 44100
    buffer_size: int = 512
    channels: int = 2
    bit_depth: int = 16
    max_file_size_mb: int = 100
    supported_formats: List[str] = field(default_factory=lambda: [
        '.wav', '.mp3', '.flac', '.ogg', '.aiff', '.m4a', '.wma'
    ])


@dataclass
class MidiConfig:
    """MIDI processing configuration"""
    resolution: int = 480
    default_tempo: float = 120.0
    default_time_signature: tuple = (4, 4)
    max_tracks: int = 64
    supported_formats: List[str] = field(default_factory=lambda: [
        '.mid', '.midi'
    ])


@dataclass
class TimingConfig:
    """Precision timing configuration"""
    microsecond_precision: bool = True
    timing_tolerance_ms: float = 0.1
    quantum_precision: bool = True
    preserve_original_timing: bool = True
    timing_correction_enabled: bool = True


@dataclass
class PluginConfig:
    """Plugin system configuration"""
    auto_discover: bool = True
    plugin_timeout_seconds: int = 30
    max_parallel_plugins: int = 4
    enable_caching: bool = True
    cache_ttl_hours: int = 24
    allowed_plugin_types: List[str] = field(default_factory=lambda: [
        'analysis', 'generation', 'arrangement', 'effects'
    ])


@dataclass
class APIConfig:
    """API server configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    workers: int = 1
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    max_upload_size_mb: int = 50
    rate_limit_per_minute: int = 60


@dataclass
class WebSocketConfig:
    """WebSocket configuration"""
    host: str = "localhost"
    port: int = 8765
    max_connections: int = 100
    heartbeat_interval: int = 30
    buffer_size: int = 65536


@dataclass
class BrainArooConfig:
    """BrainAroo AI engine configuration"""
    enable_ai_analysis: bool = True
    feature_extraction_level: str = "comprehensive"  # basic, standard, comprehensive
    max_analysis_time_seconds: int = 60
    enable_caching: bool = True
    neural_network_models: Dict[str, bool] = field(default_factory=lambda: {
        'style_classifier': True,
        'complexity_analyzer': True,
        'emotion_detector': True,
        'structure_analyzer': True
    })


@dataclass
class GenerationConfig:
    """Music generation configuration"""
    default_style: str = "pop"
    default_energy: float = 0.7
    default_complexity: float = 0.5
    max_generation_bars: int = 64
    enable_style_fusion: bool = True
    enable_ai_enhancement: bool = True
    real_time_generation: bool = False


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    file_logging: bool = True
    console_logging: bool = True
    max_log_size_mb: int = 10
    backup_count: int = 5


@dataclass
class CacheConfig:
    """Caching system configuration"""
    enabled: bool = True
    max_size_mb: int = 500
    ttl_hours: int = 24
    cleanup_interval_hours: int = 6
    cache_analysis_results: bool = True
    cache_generated_patterns: bool = True


@dataclass
class SecurityConfig:
    """Security configuration"""
    api_key_required: bool = False
    max_file_size_mb: int = 100
    allowed_file_types: List[str] = field(default_factory=lambda: [
        '.mid', '.midi', '.wav', '.mp3', '.flac', '.ogg'
    ])
    sanitize_file_names: bool = True
    validate_file_content: bool = True


class ConfigManager:
    """Central configuration manager for MuseAroo"""
    
    def __init__(self, config_file: Optional[str] = None, environment: Optional[str] = None):
        self.config_file = config_file or "musearoo_config.json"
        self.environment = Environment(environment or os.getenv('MUSEAROO_ENV', 'development'))
        
        # Initialize all config sections
        self.paths = PathConfig()
        self.audio = AudioConfig()
        self.midi = MidiConfig()
        self.timing = TimingConfig()
        self.plugins = PluginConfig()
        self.api = APIConfig()
        self.websocket = WebSocketConfig()
        self.brainaroo = BrainArooConfig()
        self.generation = GenerationConfig()
        self.logging = LoggingConfig()
        self.cache = CacheConfig()
        self.security = SecurityConfig()
        
        # Apply environment-specific overrides
        self._apply_environment_config()
        
        # Load custom config file if it exists
        self._load_config_file()
        
        # Ensure directories exist
        self.paths.ensure_directories()
        
        # Setup logging
        self._setup_logging()
    
    def _apply_environment_config(self):
        """Apply environment-specific configuration overrides"""
        
        if self.environment == Environment.DEVELOPMENT:
            self.logging.level = "DEBUG"
            self.api.reload = True
            self.plugins.auto_discover = True
            self.cache.enabled = True
            self.timing.microsecond_precision = True
            
        elif self.environment == Environment.STAGING:
            self.logging.level = "INFO"
            self.api.reload = False
            self.api.workers = 2
            self.security.api_key_required = True
            
        elif self.environment == Environment.PRODUCTION:
            self.logging.level = "WARNING"
            self.api.reload = False
            self.api.workers = 4
            self.security.api_key_required = True
            self.cache.max_size_mb = 1000
            self.plugins.max_parallel_plugins = 8
    
    def _load_config_file(self):
        """Load configuration from JSON file"""
        
        config_path = Path(self.config_file)
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                
                # Update configuration sections
                for section_name, section_data in config_data.items():
                    if hasattr(self, section_name):
                        section_obj = getattr(self, section_name)
                        for key, value in section_data.items():
                            if hasattr(section_obj, key):
                                setattr(section_obj, key, value)
                
                logging.info(f"✅ Loaded configuration from {self.config_file}")
                
            except Exception as e:
                logging.warning(f"Failed to load config file {self.config_file}: {e}")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        
        # Clear existing handlers
        logging.getLogger().handlers.clear()
        
        # Set logging level
        log_level = getattr(logging, self.logging.level.upper(), logging.INFO)
        logging.getLogger().setLevel(log_level)
        
        # Create formatter
        formatter = logging.Formatter(self.logging.format)
        
        # Console handler
        if self.logging.console_logging:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logging.getLogger().addHandler(console_handler)
        
        # File handler
        if self.logging.file_logging:
            log_file = Path(self.paths.logs_dir) / "musearoo.log"
            from logging.handlers import RotatingFileHandler
            
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=self.logging.max_log_size_mb * 1024 * 1024,
                backupCount=self.logging.backup_count
            )
            file_handler.setFormatter(formatter)
            logging.getLogger().addHandler(file_handler)
    
    def save_config(self, output_file: Optional[str] = None):
        """Save current configuration to file"""
        
        output_path = output_file or self.config_file
        
        config_data = {
            'paths': asdict(self.paths),
            'audio': asdict(self.audio),
            'midi': asdict(self.midi),
            'timing': asdict(self.timing),
            'plugins': asdict(self.plugins),
            'api': asdict(self.api),
            'websocket': asdict(self.websocket),
            'brainaroo': asdict(self.brainaroo),
            'generation': asdict(self.generation),
            'logging': asdict(self.logging),
            'cache': asdict(self.cache),
            'security': asdict(self.security)
        }
        
        try:
            with open(output_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logging.info(f"✅ Configuration saved to {output_path}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to save configuration: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        
        parts = key.split('.')
        current = self
        
        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                return default
        
        return current
    
    def set(self, key: str, value: Any) -> bool:
        """Set configuration value using dot notation"""
        
        parts = key.split('.')
        if len(parts) < 2:
            return False
        
        section_name = parts[0]
        setting_name = parts[-1]
        
        if hasattr(self, section_name):
            section = getattr(self, section_name)
            if hasattr(section, setting_name):
                setattr(section, setting_name, value)
                return True
        
        return False
    
    def validate_configuration(self) -> List[str]:
        """Validate configuration and return any issues"""
        
        issues = []
        
        # Check required directories
        for dir_name in ['data_dir', 'output_dir', 'logs_dir', 'temp_dir']:
            dir_path = getattr(self.paths, dir_name)
            if not Path(dir_path).exists():
                try:
                    Path(dir_path).mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    issues.append(f"Cannot create directory {dir_path}: {e}")
        
        # Check audio configuration
        if self.audio.sample_rate not in [22050, 44100, 48000, 96000]:
            issues.append(f"Unusual sample rate: {self.audio.sample_rate}")
        
        if self.audio.buffer_size not in [256, 512, 1024, 2048]:
            issues.append(f"Unusual buffer size: {self.audio.buffer_size}")
        
        # Check API configuration
        if not (1024 <= self.api.port <= 65535):
            issues.append(f"Invalid API port: {self.api.port}")
        
        # Check WebSocket configuration
        if not (1024 <= self.websocket.port <= 65535):
            issues.append(f"Invalid WebSocket port: {self.websocket.port}")
        
        # Check generation limits
        if self.generation.max_generation_bars > 128:
            issues.append(f"Very high max generation bars: {self.generation.max_generation_bars}")
        
        return issues
    
    def get_supported_file_types(self) -> List[str]:
        """Get all supported file types"""
        
        return list(set(
            self.audio.supported_formats + 
            self.midi.supported_formats + 
            ['.musicxml', '.xml', '.mxl']
        ))
    
    def is_file_supported(self, file_path: str) -> bool:
        """Check if file type is supported"""
        
        suffix = Path(file_path).suffix.lower()
        return suffix in self.get_supported_file_types()
    
    def get_max_file_size_bytes(self) -> int:
        """Get maximum allowed file size in bytes"""
        
        return min(
            self.audio.max_file_size_mb,
            self.security.max_file_size_mb
        ) * 1024 * 1024
    
    def __repr__(self) -> str:
        return f"ConfigManager(environment={self.environment.value}, config_file={self.config_file})"


# Global configuration instance
_config_instance: Optional[ConfigManager] = None


def get_config() -> ConfigManager:
    """Get the global configuration instance"""
    global _config_instance
    
    if _config_instance is None:
        _config_instance = ConfigManager()
    
    return _config_instance


def load_config(config_file: Optional[str] = None, environment: Optional[str] = None) -> ConfigManager:
    """Load configuration with specific parameters"""
    global _config_instance
    
    _config_instance = ConfigManager(config_file, environment)
    return _config_instance


def validate_environment() -> bool:
    """Validate the current environment setup"""
    
    config = get_config()
    issues = config.validate_configuration()
    
    if issues:
        logging.warning("Configuration issues found:")
        for issue in issues:
            logging.warning(f"  - {issue}")
        return False
    
    logging.info("✅ Environment validation successful")
    return True


def ensure_directories():
    """Ensure all required directories exist"""
    config = get_config()
    config.paths.ensure_directories()


def print_config_summary():
    """Print a summary of the current configuration"""
    
    config = get_config()
    
    print(f"""
🎼 MuseAroo Configuration Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Environment: {config.environment.value}
Config File: {config.config_file}

📁 Paths:
  Data: {config.paths.data_dir}
  Output: {config.paths.output_dir}
  Logs: {config.paths.logs_dir}
  Temp: {config.paths.temp_dir}

🎵 Audio:
  Sample Rate: {config.audio.sample_rate} Hz
  Buffer Size: {config.audio.buffer_size}
  Formats: {', '.join(config.audio.supported_formats)}

🎹 MIDI:
  Resolution: {config.midi.resolution} PPQN
  Default Tempo: {config.midi.default_tempo} BPM
  Max Tracks: {config.midi.max_tracks}

🔌 API:
  Host: {config.api.host}:{config.api.port}
  Workers: {config.api.workers}
  Reload: {config.api.reload}

🧠 BrainAroo:
  AI Analysis: {config.brainaroo.enable_ai_analysis}
  Feature Level: {config.brainaroo.feature_extraction_level}
  Max Time: {config.brainaroo.max_analysis_time_seconds}s

⚡ Timing:
  Microsecond Precision: {config.timing.microsecond_precision}
  Quantum Precision: {config.timing.quantum_precision}
  Tolerance: {config.timing.timing_tolerance_ms}ms

🔧 Plugins:
  Auto Discovery: {config.plugins.auto_discover}
  Max Parallel: {config.plugins.max_parallel_plugins}
  Timeout: {config.plugins.plugin_timeout_seconds}s

📊 Cache:
  Enabled: {config.cache.enabled}
  Max Size: {config.cache.max_size_mb}MB
  TTL: {config.cache.ttl_hours}h

🔒 Security:
  API Key Required: {config.security.api_key_required}
  Max File Size: {config.security.max_file_size_mb}MB
  File Validation: {config.security.validate_file_content}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MuseAroo Configuration Manager")
    parser.add_argument("command", choices=["show", "validate", "save", "setup"])
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--env", choices=["development", "staging", "production"], help="Environment")
    parser.add_argument("--output", help="Output file for save command")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config, args.env)
    
    if args.command == "show":
        print_config_summary()
    
    elif args.command == "validate":
        if validate_environment():
            print("✅ Configuration is valid")
        else:
            print("❌ Configuration has issues")
            exit(1)
    
    elif args.command == "save":
        if config.save_config(args.output):
            print(f"✅ Configuration saved to {args.output or config.config_file}")
        else:
            print("❌ Failed to save configuration")
            exit(1)
    
    elif args.command == "setup":
        ensure_directories()
        print("✅ Directories created")
