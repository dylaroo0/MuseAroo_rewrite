#!/usr/bin/env python3
"""
Enhanced File Handler for Music MuseAroo v2.1.0
Handles multiple audio, MIDI, and music notation formats with
intelligent detection, validation, and conversion capabilities.
"""

import os
import mimetypes
import magic
import hashlib
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

# Audio processing imports
import librosa
import soundfile as sf
import pretty_midi
from music21 import converter, stream, note, chord

# Additional format support
try:
    import mutagen
    MUTAGEN_AVAILABLE = True
except ImportError:
    MUTAGEN_AVAILABLE = False

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False


class FileType(Enum):
    """Supported file types."""
    MIDI = "midi"
    AUDIO = "audio"
    MUSICXML = "musicxml"
    UNKNOWN = "unknown"


class AudioFormat(Enum):
    """Supported audio formats."""
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    OGG = "ogg"
    M4A = "m4a"
    AIFF = "aiff"
    WMA = "wma"


class MIDIFormat(Enum):
    """Supported MIDI formats."""
    MIDI = "mid"
    MIDI_LONG = "midi"
    SMF = "smf"


class NotationFormat(Enum):
    """Supported music notation formats."""
    MUSICXML = "xml"
    MUSICXML_LONG = "musicxml"
    MXL = "mxl"
    CAPX = "capx"
    MSCZ = "mscz"
    MSCX = "mscx"


@dataclass
class FileInfo:
    """Comprehensive file information."""
    path: str
    name: str
    extension: str
    size: int
    mime_type: str
    file_type: FileType
    format_specific: Dict[str, Any]
    checksum: str
    created_at: datetime
    modified_at: datetime
    is_valid: bool
    validation_errors: List[str]
    metadata: Dict[str, Any]


@dataclass
class AudioInfo:
    """Audio file specific information."""
    duration: float
    sample_rate: int
    channels: int
    bit_depth: Optional[int]
    bitrate: Optional[int]
    codec: Optional[str]
    frames: int
    rms_energy: Optional[float]
    spectral_centroid: Optional[float]


@dataclass
class MIDIInfo:
    """MIDI file specific information."""
    duration: float
    track_count: int
    instrument_count: int
    note_count: int
    tempo_changes: List[Tuple[float, float]]
    time_signature_changes: List[Tuple[float, str]]
    key_signature_changes: List[Tuple[float, str]]
    instruments: List[Dict[str, Any]]
    has_drums: bool
    ticks_per_beat: int


@dataclass
class NotationInfo:
    """Music notation file specific information."""
    title: Optional[str]
    composer: Optional[str]
    parts_count: int
    measures_count: int
    time_signatures: List[str]
    key_signatures: List[str]
    tempos: List[float]
    instruments: List[str]


class FileHandler:
    """
    Enhanced file handler with support for multiple music file formats,
    intelligent format detection, validation, and metadata extraction.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # File type mappings
        self.extension_mappings = {
            # MIDI files
            '.mid': FileType.MIDI,
            '.midi': FileType.MIDI,
            '.smf': FileType.MIDI,
            
            # Audio files
            '.wav': FileType.AUDIO,
            '.mp3': FileType.AUDIO,
            '.flac': FileType.AUDIO,
            '.ogg': FileType.AUDIO,
            '.m4a': FileType.AUDIO,
            '.aiff': FileType.AUDIO,
            '.aif': FileType.AUDIO,
            '.wma': FileType.AUDIO,
            '.opus': FileType.AUDIO,
            '.aac': FileType.AUDIO,
            
            # Music notation files
            '.xml': FileType.MUSICXML,
            '.musicxml': FileType.MUSICXML,
            '.mxl': FileType.MUSICXML,
            '.capx': FileType.MUSICXML,
            '.mscz': FileType.MUSICXML,
            '.mscx': FileType.MUSICXML,
        }
        
        # MIME type mappings
        self.mime_mappings = {
            'audio/midi': FileType.MIDI,
            'audio/x-midi': FileType.MIDI,
            'application/x-midi': FileType.MIDI,
            'audio/wav': FileType.AUDIO,
            'audio/wave': FileType.AUDIO,
            'audio/x-wav': FileType.AUDIO,
            'audio/mpeg': FileType.AUDIO,
            'audio/mp3': FileType.AUDIO,
            'audio/flac': FileType.AUDIO,
            'audio/x-flac': FileType.AUDIO,
            'audio/ogg': FileType.AUDIO,
            'audio/vorbis': FileType.AUDIO,
            'application/xml': FileType.MUSICXML,
            'text/xml': FileType.MUSICXML,
        }
        
        # Initialize magic for file type detection
        try:
            self.magic = magic.Magic(mime=True)
            self.magic_available = True
        except:
            self.magic_available = False
            self.logger.warning("python-magic not available - using fallback detection")

    def get_file_info(self, file_path: Union[str, Path]) -> FileInfo:
        """
        Get comprehensive file information with validation and metadata extraction.
        """
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Basic file information
        stat_info = file_path.stat()
        extension = file_path.suffix.lower()
        
        # Detect file type
        file_type = self._detect_file_type(file_path)
        
        # Get MIME type
        mime_type = self._get_mime_type(file_path)
        
        # Calculate checksum
        checksum = self._calculate_checksum(file_path)
        
        # Validate file
        is_valid, validation_errors = self._validate_file(file_path, file_type)
        
        # Extract format-specific information
        format_specific = self._extract_format_info(file_path, file_type)
        
        # Extract metadata
        metadata = self._extract_metadata(file_path, file_type)
        
        return FileInfo(
            path=str(file_path),
            name=file_path.name,
            extension=extension,
            size=stat_info.st_size,
            mime_type=mime_type,
            file_type=file_type,
            format_specific=format_specific,
            checksum=checksum,
            created_at=datetime.fromtimestamp(stat_info.st_ctime),
            modified_at=datetime.fromtimestamp(stat_info.st_mtime),
            is_valid=is_valid,
            validation_errors=validation_errors,
            metadata=metadata
        )

    def _detect_file_type(self, file_path: Path) -> FileType:
        """Detect file type using multiple methods."""
        
        # 1. Try extension-based detection
        extension = file_path.suffix.lower()
        if extension in self.extension_mappings:
            file_type = self.extension_mappings[extension]
            
            # Verify with content-based detection if possible
            if self._verify_file_type_by_content(file_path, file_type):
                return file_type
        
        # 2. Try MIME type detection
        if self.magic_available:
            try:
                mime_type = self.magic.from_file(str(file_path))
                if mime_type in self.mime_mappings:
                    return self.mime_mappings[mime_type]
            except:
                pass
        
        # 3. Try content-based detection
        content_type = self._detect_by_content(file_path)
        if content_type != FileType.UNKNOWN:
            return content_type
        
        return FileType.UNKNOWN

    def _verify_file_type_by_content(self, file_path: Path, suspected_type: FileType) -> bool:
        """Verify file type by examining file content."""
        
        try:
            with open(file_path, 'rb') as f:
                header = f.read(16)
            
            if suspected_type == FileType.MIDI:
                # MIDI files start with "MThd"
                return header.startswith(b'MThd')
            
            elif suspected_type == FileType.AUDIO:
                # Check common audio file headers
                if header.startswith(b'RIFF') and b'WAVE' in header:
                    return True  # WAV
                elif header.startswith(b'ID3') or header.startswith(b'\xff\xfb'):
                    return True  # MP3
                elif header.startswith(b'fLaC'):
                    return True  # FLAC
                elif header.startswith(b'OggS'):
                    return True  # OGG
                elif header.startswith(b'FORM') and b'AIFF' in header:
                    return True  # AIFF
            
            elif suspected_type == FileType.MUSICXML:
                # XML files should start with XML declaration or have XML content
                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')[:1000]
                    return '<?xml' in content or '<score-partwise' in content or '<score-timewise' in content
                except:
                    return False
            
        except Exception as e:
            self.logger.warning(f"Content verification failed for {file_path}: {e}")
        
        return True  # Default to trusting extension if verification fails

    def _detect_by_content(self, file_path: Path) -> FileType:
        """Detect file type by examining content when extension is unreliable."""
        
        try:
            with open(file_path, 'rb') as f:
                header = f.read(16)
            
            # MIDI detection
            if header.startswith(b'MThd'):
                return FileType.MIDI
            
            # Audio detection
            if (header.startswith(b'RIFF') or 
                header.startswith(b'ID3') or 
                header.startswith(b'fLaC') or 
                header.startswith(b'OggS') or 
                header.startswith(b'FORM')):
                return FileType.AUDIO
            
            # XML detection
            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')[:1000]
                if ('<?xml' in content and 
                    ('score-partwise' in content or 'score-timewise' in content)):
                    return FileType.MUSICXML
            except:
                pass
        
        except Exception as e:
            self.logger.warning(f"Content detection failed for {file_path}: {e}")
        
        return FileType.UNKNOWN

    def _get_mime_type(self, file_path: Path) -> str:
        """Get MIME type of file."""
        
        # Try python-magic first
        if self.magic_available:
            try:
                return self.magic.from_file(str(file_path))
            except:
                pass
        
        # Fallback to mimetypes module
        mime_type, _ = mimetypes.guess_type(str(file_path))
        return mime_type or 'application/octet-stream'

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of file."""
        
        sha256_hash = hashlib.sha256()
        
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            self.logger.warning(f"Checksum calculation failed for {file_path}: {e}")
            return ""

    def _validate_file(self, file_path: Path, file_type: FileType) -> Tuple[bool, List[str]]:
        """Validate file integrity and format compliance."""
        
        errors = []
        
        try:
            if file_type == FileType.MIDI:
                errors.extend(self._validate_midi_file(file_path))
            elif file_type == FileType.AUDIO:
                errors.extend(self._validate_audio_file(file_path))
            elif file_type == FileType.MUSICXML:
                errors.extend(self._validate_musicxml_file(file_path))
        
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
        
        return len(errors) == 0, errors

    def _validate_midi_file(self, file_path: Path) -> List[str]:
        """Validate MIDI file."""
        
        errors = []
        
        try:
            midi_data = pretty_midi.PrettyMIDI(str(file_path))
            
            # Check basic structure
            if not midi_data.instruments:
                errors.append("MIDI file has no instruments")
            
            # Check for valid note data
            total_notes = sum(len(inst.notes) for inst in midi_data.instruments)
            if total_notes == 0:
                errors.append("MIDI file has no notes")
            
            # Check duration
            if midi_data.get_end_time() <= 0:
                errors.append("MIDI file has zero or negative duration")
        
        except Exception as e:
            errors.append(f"MIDI parsing error: {str(e)}")
        
        return errors

    def _validate_audio_file(self, file_path: Path) -> List[str]:
        """Validate audio file."""
        
        errors = []
        
        try:
            # Try loading with librosa
            y, sr = librosa.load(str(file_path), duration=1.0)  # Load first second
            
            if len(y) == 0:
                errors.append("Audio file is empty")
            
            if sr <= 0:
                errors.append("Invalid sample rate")
        
        except Exception as e:
            errors.append(f"Audio loading error: {str(e)}")
        
        return errors

    def _validate_musicxml_file(self, file_path: Path) -> List[str]:
        """Validate MusicXML file."""
        
        errors = []
        
        try:
            # Try parsing with music21
            score = converter.parse(str(file_path))
            
            if not score.parts:
                errors.append("MusicXML file has no parts")
            
            # Check for notes
            notes = score.flat.notes
            if not notes:
                errors.append("MusicXML file has no notes")
        
        except Exception as e:
            errors.append(f"MusicXML parsing error: {str(e)}")
        
        return errors

    def _extract_format_info(self, file_path: Path, file_type: FileType) -> Dict[str, Any]:
        """Extract format-specific information."""
        
        if file_type == FileType.MIDI:
            return self._extract_midi_info(file_path)
        elif file_type == FileType.AUDIO:
            return self._extract_audio_info(file_path)
        elif file_type == FileType.MUSICXML:
            return self._extract_notation_info(file_path)
        
        return {}

    def _extract_midi_info(self, file_path: Path) -> Dict[str, Any]:
        """Extract MIDI-specific information."""
        
        try:
            midi_data = pretty_midi.PrettyMIDI(str(file_path))
            
            # Analyze instruments
            instruments = []
            has_drums = False
            
            for inst in midi_data.instruments:
                inst_info = {
                    'name': inst.name,
                    'program': inst.program,
                    'is_drum': inst.is_drum,
                    'notes_count': len(inst.notes),
                    'channel': getattr(inst, 'channel', None)
                }
                instruments.append(inst_info)
                
                if inst.is_drum:
                    has_drums = True
            
            # Get tempo changes
            tempo_changes = []
            if hasattr(midi_data, 'get_tempo_changes'):
                times, tempos = midi_data.get_tempo_changes()
                tempo_changes = list(zip(times.tolist(), tempos.tolist()))
            
            return {
                'duration': midi_data.get_end_time(),
                'track_count': len(midi_data.instruments),
                'instrument_count': len([i for i in midi_data.instruments if not i.is_drum]),
                'note_count': sum(len(inst.notes) for inst in midi_data.instruments),
                'tempo_changes': tempo_changes,
                'time_signature_changes': self._extract_midi_time_signatures(midi_data),
                'key_signature_changes': self._extract_midi_key_signatures(midi_data),
                'instruments': instruments,
                'has_drums': has_drums,
                'ticks_per_beat': midi_data.resolution
            }
        
        except Exception as e:
            self.logger.error(f"MIDI info extraction failed: {e}")
            return {}

    def _extract_audio_info(self, file_path: Path) -> Dict[str, Any]:
        """Extract audio-specific information."""
        
        try:
            # Basic info with librosa
            y, sr = librosa.load(str(file_path), sr=None)
            duration = len(y) / sr
            
            # Try to get more detailed info with soundfile
            try:
                info = sf.info(str(file_path))
                channels = info.channels
                frames = info.frames
                bit_depth = info.subtype_info.bits if hasattr(info, 'subtype_info') else None
            except:
                channels = 1 if y.ndim == 1 else y.shape[0]
                frames = len(y)
                bit_depth = None
            
            # Calculate additional features
            rms_energy = float(librosa.feature.rms(y=y).mean()) if len(y) > 0 else None
            spectral_centroid = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean()) if len(y) > 0 else None
            
            # Try to get bitrate and codec info with mutagen
            bitrate = None
            codec = None
            
            if MUTAGEN_AVAILABLE:
                try:
                    audio_file = mutagen.File(str(file_path))
                    if audio_file is not None:
                        if hasattr(audio_file, 'info'):
                            bitrate = getattr(audio_file.info, 'bitrate', None)
                            codec = getattr(audio_file.info, 'codec', None)
                except:
                    pass
            
            return {
                'duration': duration,
                'sample_rate': int(sr),
                'channels': channels,
                'bit_depth': bit_depth,
                'bitrate': bitrate,
                'codec': codec,
                'frames': frames,
                'rms_energy': rms_energy,
                'spectral_centroid': spectral_centroid
            }
        
        except Exception as e:
            self.logger.error(f"Audio info extraction failed: {e}")
            return {}

    def _extract_notation_info(self, file_path: Path) -> Dict[str, Any]:
        """Extract music notation-specific information."""
        
        try:
            score = converter.parse(str(file_path))
            
            # Extract metadata
            metadata = score.metadata
            title = metadata.title if metadata else None
            composer = metadata.composer if metadata else None
            
            # Count parts and measures
            parts_count = len(score.parts)
            measures_count = 0
            if score.parts:
                measures_count = len(score.parts[0].getElementsByClass('Measure'))
            
            # Extract time signatures
            time_signatures = []
            for ts in score.flat.getElementsByClass('TimeSignature'):
                time_signatures.append(str(ts))
            
            # Extract key signatures
            key_signatures = []
            for ks in score.flat.getElementsByClass('KeySignature'):
                key_signatures.append(str(ks))
            
            # Extract tempos
            tempos = []
            for tempo in score.flat.getElementsByClass('TempoIndication'):
                if hasattr(tempo, 'number'):
                    tempos.append(float(tempo.number))
            
            # Extract instruments
            instruments = []
            for part in score.parts:
                if part.partName:
                    instruments.append(part.partName)
                elif hasattr(part, 'instrumentName'):
                    instruments.append(part.instrumentName)
            
            return {
                'title': title,
                'composer': composer,
                'parts_count': parts_count,
                'measures_count': measures_count,
                'time_signatures': time_signatures,
                'key_signatures': key_signatures,
                'tempos': tempos,
                'instruments': instruments
            }
        
        except Exception as e:
            self.logger.error(f"Notation info extraction failed: {e}")
            return {}

    def _extract_midi_time_signatures(self, midi_data) -> List[Tuple[float, str]]:
        """Extract time signature changes from MIDI."""
        
        time_sigs = []
        try:
            if hasattr(midi_data, 'time_signature_changes'):
                for ts in midi_data.time_signature_changes:
                    time_sigs.append((ts.time, f"{ts.numerator}/{ts.denominator}"))
        except:
            pass
        
        return time_sigs

    def _extract_midi_key_signatures(self, midi_data) -> List[Tuple[float, str]]:
        """Extract key signature changes from MIDI."""
        
        key_sigs = []
        try:
            if hasattr(midi_data, 'key_signature_changes'):
                for ks in midi_data.key_signature_changes:
                    key_sigs.append((ks.time, f"{ks.key_number}"))
        except:
            pass
        
        return key_sigs

    def _extract_metadata(self, file_path: Path, file_type: FileType) -> Dict[str, Any]:
        """Extract general metadata from file."""
        
        metadata = {}
        
        if MUTAGEN_AVAILABLE and file_type == FileType.AUDIO:
            try:
                audio_file = mutagen.File(str(file_path))
                if audio_file is not None:
                    # Common tags
                    tag_mappings = {
                        'TIT2': 'title',
                        'TPE1': 'artist',
                        'TALB': 'album',
                        'TDRC': 'year',
                        'TCON': 'genre',
                        'TRCK': 'track',
                        # Alternative tag names
                        'TITLE': 'title',
                        'ARTIST': 'artist',
                        'ALBUM': 'album',
                        'DATE': 'year',
                        'GENRE': 'genre',
                    }
                    
                    for tag_key, meta_key in tag_mappings.items():
                        if tag_key in audio_file:
                            value = audio_file[tag_key]
                            if isinstance(value, list):
                                value = value[0] if value else None
                            if value:
                                metadata[meta_key] = str(value)
            except:
                pass
        
        return metadata

    def convert_audio_format(
        self, 
        input_path: Union[str, Path], 
        output_path: Union[str, Path],
        target_format: str = "wav",
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None
    ) -> bool:
        """Convert audio file to different format."""
        
        try:
            input_path = Path(input_path)
            output_path = Path(output_path)
            
            # Load audio
            y, sr = librosa.load(str(input_path), sr=sample_rate)
            
            # Convert channels if needed
            if channels and channels != (1 if y.ndim == 1 else y.shape[0]):
                if channels == 1 and y.ndim > 1:
                    y = librosa.to_mono(y)
                elif channels == 2 and y.ndim == 1:
                    y = np.array([y, y])
            
            # Save in target format
            sf.write(str(output_path), y, sr, format=target_format.upper())
            
            self.logger.info(f"Converted {input_path} to {output_path}")
            return True
        
        except Exception as e:
            self.logger.error(f"Audio conversion failed: {e}")
            return False

    def extract_audio_features(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Extract comprehensive audio features for analysis."""
        
        try:
            y, sr = librosa.load(str(file_path))
            
            features = {}
            
            # Basic features
            features['duration'] = len(y) / sr
            features['sample_rate'] = sr
            features['rms_energy'] = float(librosa.feature.rms(y=y).mean())
            
            # Spectral features
            features['spectral_centroid'] = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
            features['spectral_rolloff'] = float(librosa.feature.spectral_rolloff(y=y, sr=sr).mean())
            features['spectral_bandwidth'] = float(librosa.feature.spectral_bandwidth(y=y, sr=sr).mean())
            features['zero_crossing_rate'] = float(librosa.feature.zero_crossing_rate(y).mean())
            
            # Tempo and rhythm
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            features['tempo'] = float(tempo)
            features['beat_count'] = len(beats)
            
            # Harmony
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features['chroma_mean'] = chroma.mean(axis=1).tolist()
            
            # MFCCs
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features['mfcc_mean'] = mfcc.mean(axis=1).tolist()
            
            return features
        
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            return {}

    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Get list of supported file formats by category."""
        
        return {
            'midi': [ext for ext, ftype in self.extension_mappings.items() if ftype == FileType.MIDI],
            'audio': [ext for ext, ftype in self.extension_mappings.items() if ftype == FileType.AUDIO],
            'notation': [ext for ext, ftype in self.extension_mappings.items() if ftype == FileType.MUSICXML]
        }

    def is_supported_file(self, file_path: Union[str, Path]) -> bool:
        """Check if file is supported."""
        
        extension = Path(file_path).suffix.lower()
        return extension in self.extension_mappings

    def batch_process_directory(
        self, 
        directory: Union[str, Path], 
        pattern: str = "*",
        recursive: bool = False
    ) -> List[FileInfo]:
        """Process all supported files in a directory."""
        
        directory = Path(directory)
        results = []
        
        if recursive:
            files = directory.rglob(pattern)
        else:
            files = directory.glob(pattern)
        
        for file_path in files:
            if file_path.is_file() and self.is_supported_file(file_path):
                try:
                    file_info = self.get_file_info(file_path)
                    results.append(file_info)
                except Exception as e:
                    self.logger.error(f"Failed to process {file_path}: {e}")
        
        return results


# Utility functions
def get_file_handler() -> FileHandler:
    """Get file handler instance."""
    return FileHandler()


def quick_file_check(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Quick file check without full analysis."""
    
    handler = FileHandler()
    file_path = Path(file_path)
    
    return {
        'exists': file_path.exists(),
        'size': file_path.stat().st_size if file_path.exists() else 0,
        'extension': file_path.suffix.lower(),
        'supported': handler.is_supported_file(file_path),
        'detected_type': handler._detect_file_type(file_path).value if file_path.exists() else 'unknown'
    }


# CLI interface
if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Music MuseAroo File Handler")
    parser.add_argument("command", choices=["info", "validate", "convert", "batch", "formats"])
    parser.add_argument("--input", help="Input file or directory path")
    parser.add_argument("--output", help="Output file path (for convert)")
    parser.add_argument("--format", help="Target format (for convert)")
    parser.add_argument("--recursive", action="store_true", help="Recursive directory processing")
    
    args = parser.parse_args()
    
    handler = FileHandler()
    
    if args.command == "info":
        if not args.input:
            print("Error: --input required for info command")
            exit(1)
        
        file_info = handler.get_file_info(args.input)
        print(json.dumps(file_info.__dict__, indent=2, default=str))
    
    elif args.command == "validate":
        if not args.input:
            print("Error: --input required for validate command")
            exit(1)
        
        file_info = handler.get_file_info(args.input)
        if file_info.is_valid:
            print("✅ File is valid")
        else:
            print("❌ File validation failed:")
            for error in file_info.validation_errors:
                print(f"   - {error}")
    
    elif args.command == "convert":
        if not all([args.input, args.output, args.format]):
            print("Error: --input, --output, and --format required for convert")
            exit(1)
        
        success = handler.convert_audio_format(args.input, args.output, args.format)
        if success:
            print("✅ Conversion successful")
        else:
            print("❌ Conversion failed")
    
    elif args.command == "batch":
        if not args.input:
            print("Error: --input required for batch command")
            exit(1)
        
        results = handler.batch_process_directory(args.input, recursive=args.recursive)
        print(f"Processed {len(results)} files:")
        for result in results:
            print(f"   {result.name} - {result.file_type.value} - {'✅' if result.is_valid else '❌'}")
    
    elif args.command == "formats":
        formats = handler.get_supported_formats()
        print("Supported formats:")
        for category, extensions in formats.items():
            print(f"  {category}: {', '.join(extensions)}")
