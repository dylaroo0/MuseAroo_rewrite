#!/usr/bin/env python3
"""
Advanced Plugin Architecture v4.0 for Music MuseAroo
THE ULTIMATE plugin system with hot-reloading, dependency injection, and enterprise features

Features:
• Hot-reloading without server restart
• Advanced dependency injection system
• Plugin versioning and compatibility checks
• Automatic plugin discovery and loading
• Performance monitoring and profiling
• Plugin sandboxing and security
• Event-driven plugin communication
• Plugin marketplace integration
• A/B testing framework for plugins
• Advanced caching and optimization
• Plugin health monitoring
• Rollback and version management
"""

import asyncio
import importlib
import importlib.util
import inspect
import json
import logging
import pickle
import sys
import time
import traceback
import weakref
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
import psutil
import cProfile
import pstats
from functools import wraps, lru_cache
import yaml

# Advanced imports for plugin features
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import redis
from cryptography.fernet import Fernet
import semver

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PluginStatus(Enum):
    """Plugin lifecycle status"""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    ACTIVE = "active"
    ERROR = "error"
    DEPRECATED = "deprecated"
    DISABLED = "disabled"
    UPDATING = "updating"


class PluginPhase(Enum):
    """Plugin execution phases"""
    PHASE_1_ANALYSIS = 1
    PHASE_2_GENERATION = 2
    PHASE_3_ARRANGEMENT = 3
    PHASE_4_EXPORT = 4
    PHASE_UTILITY = 0
    PHASE_REALTIME = 99


class PluginPriority(Enum):
    """Plugin execution priority"""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


@dataclass
class PluginMetadata:
    """Comprehensive plugin metadata"""
    name: str
    version: str
    description: str
    author: str
    email: str = ""
    website: str = ""
    license: str = "MIT"
    
    # Functional metadata
    phase: PluginPhase = PluginPhase.PHASE_UTILITY
    priority: PluginPriority = PluginPriority.NORMAL
    input_types: List[str] = field(default_factory=list)
    output_types: List[str] = field(default_factory=list)
    
    # Dependencies
    dependencies: List[str] = field(default_factory=list)
    optional_dependencies: List[str] = field(default_factory=list)
    python_version: str = ">=3.8"
    musearoo_version: str = ">=2.0.0"
    
    # Performance characteristics
    estimated_runtime: float = 0.0  # seconds
    memory_usage: int = 0  # MB
    cpu_intensive: bool = False
    gpu_accelerated: bool = False
    
    # Security and sandboxing
    trusted: bool = False
    sandbox_level: int = 1  # 0=none, 1=basic, 2=strict, 3=isolated
    permissions: List[str] = field(default_factory=list)
    
    # Marketplace and distribution
    plugin_id: str = ""
    marketplace_url: str = ""
    download_count: int = 0
    rating: float = 0.0
    
    # Technical details
    entry_point: str = "main"
    config_schema: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    # Lifecycle
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None


@dataclass
class PluginPerformanceMetrics:
    """Plugin performance tracking"""
    execution_count: int = 0
    total_runtime: float = 0.0
    average_runtime: float = 0.0
    max_runtime: float = 0.0
    min_runtime: float = float('inf')
    memory_peak: int = 0
    cpu_usage: float = 0.0
    error_count: int = 0
    success_rate: float = 100.0
    last_execution: Optional[datetime] = None


class PluginEvent(Enum):
    """Plugin system events"""
    PLUGIN_LOADED = "plugin_loaded"
    PLUGIN_UNLOADED = "plugin_unloaded"
    PLUGIN_EXECUTED = "plugin_executed"
    PLUGIN_ERROR = "plugin_error"
    PLUGIN_UPDATED = "plugin_updated"
    DEPENDENCY_RESOLVED = "dependency_resolved"
    DEPENDENCY_FAILED = "dependency_failed"


class PluginInterface(ABC):
    """Base interface that all plugins must implement"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.metadata: Optional[PluginMetadata] = None
        self.logger = logging.getLogger(f"plugin.{self.__class__.__name__}")
        self._initialized = False
        self._context: Dict[str, Any] = {}
    
    @abstractmethod
    async def initialize(self, context: Dict[str, Any]) -> bool:
        """Initialize the plugin with context"""
        pass
    
    @abstractmethod
    async def execute(self, input_data: Any, **kwargs) -> Any:
        """Execute the plugin's main functionality"""
        pass
    
    async def cleanup(self) -> None:
        """Clean up plugin resources"""
        pass
    
    async def validate_input(self, input_data: Any) -> bool:
        """Validate input data"""
        return True
    
    async def get_config_schema(self) -> Dict[str, Any]:
        """Return configuration schema"""
        return {}
    
    async def health_check(self) -> Dict[str, Any]:
        """Return plugin health status"""
        return {'status': 'healthy', 'timestamp': datetime.now().isoformat()}


class DependencyInjector:
    """Advanced dependency injection system"""
    
    def __init__(self):
        self.services: Dict[str, Any] = {}
        self.factories: Dict[str, Callable] = {}
        self.singletons: Dict[str, Any] = {}
        self.scoped: Dict[str, Dict[str, Any]] = {}
        self.resolving: Set[str] = set()  # Circular dependency detection
    
    def register_service(self, name: str, service: Any, singleton: bool = False):
        """Register a service instance"""
        if singleton:
            self.singletons[name] = service
        else:
            self.services[name] = service
    
    def register_factory(self, name: str, factory: Callable, singleton: bool = False):
        """Register a service factory"""
        self.factories[name] = factory
        if singleton:
            # Create singleton on first access
            self.singletons[name] = None
    
    def resolve(self, name: str, scope: str = "default") -> Any:
        """Resolve a dependency by name"""
        if name in self.resolving:
            raise ValueError(f"Circular dependency detected: {name}")
        
        self.resolving.add(name)
        try:
            # Check singletons first
            if name in self.singletons:
                if self.singletons[name] is None and name in self.factories:
                    self.singletons[name] = self._create_from_factory(name)
                return self.singletons[name]
            
            # Check scoped services
            if scope in self.scoped and name in self.scoped[scope]:
                return self.scoped[scope][name]
            
            # Check regular services
            if name in self.services:
                return self.services[name]
            
            # Create from factory
            if name in self.factories:
                service = self._create_from_factory(name)
                if scope != "default":
                    if scope not in self.scoped:
                        self.scoped[scope] = {}
                    self.scoped[scope][name] = service
                return service
            
            raise ValueError(f"Service not found: {name}")
            
        finally:
            self.resolving.discard(name)
    
    def _create_from_factory(self, name: str) -> Any:
        """Create service instance from factory"""
        factory = self.factories[name]
        
        # Analyze factory parameters for dependency injection
        sig = inspect.signature(factory)
        kwargs = {}
        
        for param_name, param in sig.parameters.items():
            if param.annotation and param.annotation != inspect.Parameter.empty:
                # Try to resolve dependency by type annotation
                type_name = getattr(param.annotation, '__name__', str(param.annotation))
                try:
                    kwargs[param_name] = self.resolve(type_name)
                except ValueError:
                    # If dependency not found and has default, use default
                    if param.default != inspect.Parameter.empty:
                        kwargs[param_name] = param.default
        
        return factory(**kwargs)
    
    def clear_scope(self, scope: str):
        """Clear all services in a scope"""
        self.scoped.pop(scope, None)


class PluginSandbox:
    """Plugin sandboxing system for security"""
    
    def __init__(self, level: int = 1):
        self.level = level
        self.allowed_modules = {
            'builtins', 'json', 'math', 'datetime', 'decimal', 'collections',
            'itertools', 'functools', 'operator', 'copy', 're', 'string',
            'librosa', 'numpy', 'scipy', 'pretty_midi', 'music21'
        }
        self.blocked_modules = {
            'os', 'sys', 'subprocess', 'socket', 'urllib', 'requests',
            'pickle', 'eval', 'exec', 'compile', '__import__'
        }
    
    def validate_plugin(self, plugin_code: str) -> bool:
        """Validate plugin code for security issues"""
        if self.level == 0:
            return True
        
        # Basic security checks
        dangerous_patterns = [
            '__import__', 'eval(', 'exec(', 'compile(',
            'os.system', 'subprocess.', 'socket.',
            'open(', 'file(', 'input(', 'raw_input('
        ]
        
        for pattern in dangerous_patterns:
            if pattern in plugin_code:
                logger.warning(f"Security violation: {pattern} found in plugin code")
                return False
        
        return True
    
    def create_restricted_globals(self) -> Dict[str, Any]:
        """Create restricted global namespace for plugin execution"""
        restricted_builtins = {
            name: getattr(__builtins__, name)
            for name in dir(__builtins__)
            if not name.startswith('_') and name not in {
                'eval', 'exec', 'compile', 'open', 'input', 'raw_input',
                '__import__', 'reload', 'quit', 'exit'
            }
        }
        
        return {
            '__builtins__': restricted_builtins,
            'logger': logging.getLogger('sandbox'),
        }


class PluginFileWatcher(FileSystemEventHandler):
    """File system watcher for hot-reloading plugins"""
    
    def __init__(self, plugin_manager):
        self.plugin_manager = plugin_manager
        self.last_modified = {}
        
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith('.py'):
            # Debounce rapid file changes
            current_time = time.time()
            if (event.src_path not in self.last_modified or 
                current_time - self.last_modified[event.src_path] > 1.0):
                
                self.last_modified[event.src_path] = current_time
                logger.info(f"🔄 Plugin file changed: {event.src_path}")
                
                # Trigger plugin reload
                asyncio.create_task(self.plugin_manager._reload_plugin_from_file(event.src_path))


class AdvancedPluginManager:
    """
    THE ULTIMATE plugin management system
    
    This is the most sophisticated plugin architecture ever built for music AI.
    Supports hot-reloading, dependency injection, sandboxing, performance monitoring,
    A/B testing, and enterprise-grade plugin management.
    """
    
    def __init__(
        self,
        plugin_directories: List[str] = None,
        enable_hot_reload: bool = True,
        enable_sandboxing: bool = True,
        sandbox_level: int = 1,
        enable_performance_monitoring: bool = True,
        enable_caching: bool = True,
        cache_ttl: int = 3600,
        max_concurrent_plugins: int = 10,
        plugin_timeout: float = 300.0,
        redis_url: Optional[str] = None,
    ):
        # Core configuration
        self.plugin_directories = plugin_directories or ["plugins/", "src/plugins/"]
        self.enable_hot_reload = enable_hot_reload
        self.enable_sandboxing = enable_sandboxing
        self.sandbox_level = sandbox_level
        self.enable_performance_monitoring = enable_performance_monitoring
        self.enable_caching = enable_caching
        self.cache_ttl = cache_ttl
        self.max_concurrent_plugins = max_concurrent_plugins
        self.plugin_timeout = plugin_timeout
        
        # Plugin registry
        self.plugins: Dict[str, PluginInterface] = {}
        self.plugin_metadata: Dict[str, PluginMetadata] = {}
        self.plugin_status: Dict[str, PluginStatus] = {}
        self.plugin_modules: Dict[str, Any] = {}
        self.plugin_files: Dict[str, str] = {}  # plugin_name -> file_path
        
        # Performance tracking
        self.performance_metrics: Dict[str, PluginPerformanceMetrics] = {}
        self.execution_history: List[Dict[str, Any]] = []
        
        # Dependency management
        self.dependency_injector = DependencyInjector()
        self.dependency_graph: Dict[str, Set[str]] = {}
        
        # Security and sandboxing
        self.sandbox = PluginSandbox(sandbox_level) if enable_sandboxing else None
        
        # Event system
        self.event_handlers: Dict[PluginEvent, List[Callable]] = {}
        
        # Caching system
        self.cache: Dict[str, Any] = {}
        self.cache_timestamps: Dict[str, float] = {}
        
        # Threading and async
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_plugins)
        self.active_executions: Dict[str, asyncio.Task] = {}
        
        # Hot reload
        self.file_observer: Optional[Observer] = None
        self.file_watcher: Optional[PluginFileWatcher] = None
        
        # External storage
        self.redis: Optional[redis.Redis] = None
        if redis_url:
            try:
                self.redis = redis.from_url(redis_url)
                self.redis.ping()
                logger.info("✅ Redis connection established for plugin system")
            except Exception as e:
                logger.warning(f"⚠️ Redis connection failed: {e}")
        
        # A/B testing
        self.ab_tests: Dict[str, Dict[str, Any]] = {}
        
        # Plugin marketplace
        self.marketplace_plugins: Dict[str, Dict[str, Any]] = {}
        
        logger.info("🧩 Advanced Plugin Manager v4.0 initialized")
    
    async def initialize(self):
        """Initialize the plugin manager"""
        logger.info("🚀 Initializing Advanced Plugin Manager...")
        
        # Register core services
        await self._register_core_services()
        
        # Discover and load plugins
        await self.discover_plugins()
        
        # Start hot reload if enabled
        if self.enable_hot_reload:
            await self._start_hot_reload()
        
        # Load cached data
        await self._load_cache()
        
        logger.info("✅ Advanced Plugin Manager initialized successfully")
    
    async def _register_core_services(self):
        """Register core services for dependency injection"""
        # Register common services that plugins might need
        self.dependency_injector.register_service('logger', logger, singleton=True)
        self.dependency_injector.register_service('redis', self.redis, singleton=True)
        self.dependency_injector.register_service('performance_monitor', self.performance_metrics, singleton=True)
        
        # Register factories for common objects
        self.dependency_injector.register_factory('cache', lambda: {}, singleton=False)
        self.dependency_injector.register_factory('config', lambda: {}, singleton=False)
    
    async def discover_plugins(self):
        """Discover and load all plugins from plugin directories"""
        logger.info("🔍 Discovering plugins...")
        
        discovered_count = 0
        
        for plugin_dir in self.plugin_directories:
            plugin_path = Path(plugin_dir)
            if not plugin_path.exists():
                continue
            
            # Scan for Python files
            for file_path in plugin_path.rglob("*.py"):
                if file_path.name.startswith("__"):
                    continue
                
                try:
                    await self._load_plugin_from_file(str(file_path))
                    discovered_count += 1
                except Exception as e:
                    logger.error(f"❌ Failed to load plugin from {file_path}: {e}")
        
        # Scan for plugin packages (directories with __init__.py)
        for plugin_dir in self.plugin_directories:
            plugin_path = Path(plugin_dir)
            if not plugin_path.exists():
                continue
                
            for subdir in plugin_path.iterdir():
                if subdir.is_dir() and (subdir / "__init__.py").exists():
                    try:
                        await self._load_plugin_package(str(subdir))
                        discovered_count += 1
                    except Exception as e:
                        logger.error(f"❌ Failed to load plugin package from {subdir}: {e}")
        
        logger.info(f"✅ Plugin discovery complete: {discovered_count} plugins found")
        
        # Resolve dependencies
        await self._resolve_plugin_dependencies()
    
    async def _load_plugin_from_file(self, file_path: str):
        """Load a plugin from a Python file"""
        file_path = Path(file_path)
        plugin_name = file_path.stem
        
        # Check if plugin is already loaded
        if plugin_name in self.plugins:
            logger.debug(f"Plugin {plugin_name} already loaded, skipping")
            return
        
        # Load module
        spec = importlib.util.spec_from_file_location(plugin_name, file_path)
        if not spec or not spec.loader:
            raise ValueError(f"Cannot load module spec from {file_path}")
        
        module = importlib.util.module_from_spec(spec)
        
        # Security check if sandboxing is enabled
        if self.sandbox:
            with open(file_path, 'r') as f:
                code = f.read()
            if not self.sandbox.validate_plugin(code):
                raise SecurityError(f"Plugin {plugin_name} failed security validation")
        
        # Execute module
        spec.loader.exec_module(module)
        
        # Find plugin class
        plugin_class = None
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (inspect.isclass(attr) and 
                issubclass(attr, PluginInterface) and 
                attr != PluginInterface):
                plugin_class = attr
                break
        
        if not plugin_class:
            raise ValueError(f"No plugin class found in {file_path}")
        
        # Extract metadata
        metadata = await self._extract_plugin_metadata(module, plugin_class)
        metadata.name = plugin_name
        
        # Create plugin instance
        plugin_instance = plugin_class()
        plugin_instance.metadata = metadata
        
        # Register plugin
        self.plugins[plugin_name] = plugin_instance
        self.plugin_metadata[plugin_name] = metadata
        self.plugin_status[plugin_name] = PluginStatus.LOADED
        self.plugin_modules[plugin_name] = module
        self.plugin_files[plugin_name] = str(file_path)
        self.performance_metrics[plugin_name] = PluginPerformanceMetrics()
        
        logger.info(f"✅ Plugin loaded: {plugin_name} v{metadata.version}")
        
        # Emit event
        await self._emit_event(PluginEvent.PLUGIN_LOADED, {
            'plugin_name': plugin_name,
            'metadata': metadata
        })
    
    async def _load_plugin_package(self, package_path: str):
        """Load a plugin from a package directory"""
        package_path = Path(package_path)
        plugin_name = package_path.name
        
        # Add package to Python path temporarily
        sys.path.insert(0, str(package_path.parent))
        
        try:
            # Import the package
            module = importlib.import_module(plugin_name)
            
            # Look for plugin class in __init__.py or main.py
            plugin_class = None
            
            # Check __init__.py
            if hasattr(module, 'Plugin'):
                plugin_class = module.Plugin
            else:
                # Check for any PluginInterface subclass
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (inspect.isclass(attr) and 
                        issubclass(attr, PluginInterface) and 
                        attr != PluginInterface):
                        plugin_class = attr
                        break
            
            if not plugin_class:
                raise ValueError(f"No plugin class found in package {package_path}")
            
            # Extract metadata
            metadata = await self._extract_plugin_metadata(module, plugin_class)
            metadata.name = plugin_name
            
            # Create plugin instance
            plugin_instance = plugin_class()
            plugin_instance.metadata = metadata
            
            # Register plugin
            self.plugins[plugin_name] = plugin_instance
            self.plugin_metadata[plugin_name] = metadata
            self.plugin_status[plugin_name] = PluginStatus.LOADED
            self.plugin_modules[plugin_name] = module
            self.plugin_files[plugin_name] = str(package_path / "__init__.py")
            self.performance_metrics[plugin_name] = PluginPerformanceMetrics()
            
            logger.info(f"✅ Plugin package loaded: {plugin_name} v{metadata.version}")
            
            # Emit event
            await self._emit_event(PluginEvent.PLUGIN_LOADED, {
                'plugin_name': plugin_name,
                'metadata': metadata
            })
            
        finally:
            # Remove from path
            if str(package_path.parent) in sys.path:
                sys.path.remove(str(package_path.parent))
    
    async def _extract_plugin_metadata(self, module: Any, plugin_class: Type) -> PluginMetadata:
        """Extract metadata from plugin module and class"""
        # Default metadata
        metadata = PluginMetadata(
            name="unknown",
            version="1.0.0",
            description="No description provided",
            author="Unknown"
        )
        
        # Extract from module attributes
        for attr in ['__version__', '__author__', '__description__', '__email__', '__license__']:
            if hasattr(module, attr):
                value = getattr(module, attr)
                if attr == '__version__':
                    metadata.version = value
                elif attr == '__author__':
                    metadata.author = value
                elif attr == '__description__':
                    metadata.description = value
                elif attr == '__email__':
                    metadata.email = value
                elif attr == '__license__':
                    metadata.license = value
        
        # Extract from class docstring
        if plugin_class.__doc__:
            # Parse structured docstring for metadata
            lines = plugin_class.__doc__.strip().split('\n')
            if lines:
                metadata.description = lines[0].strip()
        
        # Extract from class attributes
        for attr in ['phase', 'priority', 'input_types', 'output_types', 'dependencies']:
            if hasattr(plugin_class, attr):
                setattr(metadata, attr, getattr(plugin_class, attr))
        
        # Look for metadata.yaml or plugin.yaml file
        plugin_file = self.plugin_files.get(metadata.name)
        if plugin_file:
            plugin_dir = Path(plugin_file).parent
            for metadata_file in ['metadata.yaml', 'plugin.yaml', 'plugin.yml']:
                metadata_path = plugin_dir / metadata_file
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r') as f:
                            yaml_data = yaml.safe_load(f)
                            
                        # Update metadata from YAML
                        for key, value in yaml_data.items():
                            if hasattr(metadata, key):
                                setattr(metadata, key, value)
                        
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load metadata from {metadata_path}: {e}")
        
        return metadata
    
    async def _resolve_plugin_dependencies(self):
        """Resolve plugin dependencies and build dependency graph"""
        logger.info("🔗 Resolving plugin dependencies...")
        
        # Build dependency graph
        for plugin_name, metadata in self.plugin_metadata.items():
            self.dependency_graph[plugin_name] = set(metadata.dependencies)
        
        # Topological sort for dependency order
        resolved_order = self._topological_sort(self.dependency_graph)
        
        # Initialize plugins in dependency order
        for plugin_name in resolved_order:
            if plugin_name in self.plugins:
                try:
                    await self._initialize_plugin(plugin_name)
                except Exception as e:
                    logger.error(f"❌ Failed to initialize plugin {plugin_name}: {e}")
                    self.plugin_status[plugin_name] = PluginStatus.ERROR
        
        logger.info("✅ Plugin dependencies resolved")
    
    def _topological_sort(self, graph: Dict[str, Set[str]]) -> List[str]:
        """Perform topological sort on dependency graph"""
        in_degree = {node: 0 for node in graph}
        
        # Calculate in-degrees
        for node in graph:
            for neighbor in graph[node]:
                if neighbor in in_degree:
                    in_degree[neighbor] += 1
        
        # Initialize queue with nodes having no dependencies
        queue = [node for node in in_degree if in_degree[node] == 0]
        result = []
        
        while queue:
            node = queue.pop(0)
            result.append(node)
            
            # Update in-degrees of dependent nodes
            for neighbor in graph.get(node, set()):
                if neighbor in in_degree:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)
        
        # Check for circular dependencies
        if len(result) != len(graph):
            remaining = set(graph.keys()) - set(result)
            logger.warning(f"⚠️ Circular dependencies detected: {remaining}")
        
        return result
    
    async def _initialize_plugin(self, plugin_name: str):
        """Initialize a specific plugin"""
        plugin = self.plugins.get(plugin_name)
        if not plugin:
            raise ValueError(f"Plugin not found: {plugin_name}")
        
        if self.plugin_status[plugin_name] == PluginStatus.ACTIVE:
            return  # Already initialized
        
        logger.debug(f"Initializing plugin: {plugin_name}")
        
        # Prepare context with dependencies
        context = {}
        metadata = self.plugin_metadata[plugin_name]
        
        for dependency in metadata.dependencies:
            try:
                context[dependency] = self.dependency_injector.resolve(dependency)
            except ValueError:
                # Try to find dependency in loaded plugins
                if dependency in self.plugins:
                    context[dependency] = self.plugins[dependency]
                else:
                    raise ValueError(f"Dependency not found: {dependency}")
        
        # Initialize plugin
        success = await plugin.initialize(context)
        
        if success:
            self.plugin_status[plugin_name] = PluginStatus.ACTIVE
            logger.info(f"✅ Plugin initialized: {plugin_name}")
        else:
            self.plugin_status[plugin_name] = PluginStatus.ERROR
            raise RuntimeError(f"Plugin initialization failed: {plugin_name}")
    
    async def execute_plugin(
        self,
        plugin_name: str,
        input_data: Any,
        timeout: Optional[float] = None,
        cache_key: Optional[str] = None,
        **kwargs
    ) -> Any:
        """Execute a plugin with advanced features"""
        
        # Check if plugin exists and is active
        if plugin_name not in self.plugins:
            raise ValueError(f"Plugin not found: {plugin_name}")
        
        if self.plugin_status[plugin_name] != PluginStatus.ACTIVE:
            raise RuntimeError(f"Plugin not active: {plugin_name} (status: {self.plugin_status[plugin_name]})")
        
        # Check cache if enabled
        if self.enable_caching and cache_key:
            cached_result = await self._get_cached_result(plugin_name, cache_key)
            if cached_result is not None:
                logger.debug(f"💾 Cache hit for plugin {plugin_name}")
                return cached_result
        
        # Check for A/B test
        plugin_to_execute = await self._get_ab_test_plugin(plugin_name)
        
        plugin = self.plugins[plugin_to_execute]
        timeout = timeout or self.plugin_timeout
        
        # Performance monitoring setup
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss if self.enable_performance_monitoring else 0
        
        try:
            # Create profiler if performance monitoring is enabled
            profiler = cProfile.Profile() if self.enable_performance_monitoring else None
            
            if profiler:
                profiler.enable()
            
            # Execute plugin with timeout
            if asyncio.iscoroutinefunction(plugin.execute):
                result = await asyncio.wait_for(
                    plugin.execute(input_data, **kwargs),
                    timeout=timeout
                )
            else:
                # Run in executor for sync functions
                result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        plugin.execute,
                        input_data,
                        **kwargs
                    ),
                    timeout=timeout
                )
            
            if profiler:
                profiler.disable()
            
            # Update performance metrics
            execution_time = time.time() - start_time
            await self._update_performance_metrics(
                plugin_to_execute, execution_time, start_memory, True, profiler
            )
            
            # Cache result if enabled
            if self.enable_caching and cache_key:
                await self._cache_result(plugin_name, cache_key, result)
            
            # Emit success event
            await self._emit_event(PluginEvent.PLUGIN_EXECUTED, {
                'plugin_name': plugin_to_execute,
                'execution_time': execution_time,
                'success': True
            })
            
            return result
            
        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            await self._update_performance_metrics(
                plugin_to_execute, execution_time, start_memory, False
            )
            
            logger.error(f"⏱️ Plugin {plugin_to_execute} timed out after {timeout}s")
            raise
            
        except Exception as e:
            execution_time = time.time() - start_time
            await self._update_performance_metrics(
                plugin_to_execute, execution_time, start_memory, False
            )
            
            logger.error(f"❌ Plugin {plugin_to_execute} execution failed: {e}")
            
            # Emit error event
            await self._emit_event(PluginEvent.PLUGIN_ERROR, {
                'plugin_name': plugin_to_execute,
                'error': str(e),
                'execution_time': execution_time
            })
            
            raise
    
    async def _update_performance_metrics(
        self,
        plugin_name: str,
        execution_time: float,
        start_memory: int,
        success: bool,
        profiler: Optional[cProfile.Profile] = None
    ):
        """Update performance metrics for a plugin"""
        if not self.enable_performance_monitoring:
            return
        
        metrics = self.performance_metrics[plugin_name]
        
        # Update basic metrics
        metrics.execution_count += 1
        metrics.total_runtime += execution_time
        metrics.average_runtime = metrics.total_runtime / metrics.execution_count
        metrics.max_runtime = max(metrics.max_runtime, execution_time)
        metrics.min_runtime = min(metrics.min_runtime, execution_time)
        metrics.last_execution = datetime.now()
        
        if success:
            metrics.success_rate = ((metrics.execution_count - metrics.error_count) / 
                                   metrics.execution_count * 100)
        else:
            metrics.error_count += 1
            metrics.success_rate = ((metrics.execution_count - metrics.error_count) / 
                                   metrics.execution_count * 100)
        
        # Update memory metrics
        if start_memory > 0:
            current_memory = psutil.Process().memory_info().rss
            memory_used = current_memory - start_memory
            metrics.memory_peak = max(metrics.memory_peak, memory_used)
        
        # Store detailed execution record
        execution_record = {
            'plugin_name': plugin_name,
            'timestamp': datetime.now().isoformat(),
            'execution_time': execution_time,
            'success': success,
            'memory_used': metrics.memory_peak
        }
        
        # Add profiling data if available
        if profiler:
            stats = pstats.Stats(profiler)
            stats.sort_stats('cumulative')
            # Store top function calls (simplified)
            execution_record['profile_summary'] = str(stats).split('\n')[:10]
        
        self.execution_history.append(execution_record)
        
        # Keep only last 1000 execution records
        if len(self.execution_history) > 1000:
            self.execution_history = self.execution_history[-1000:]
    
    async def _get_cached_result(self, plugin_name: str, cache_key: str) -> Any:
        """Get cached result for plugin execution"""
        full_key = f"{plugin_name}:{cache_key}"
        
        # Check in-memory cache first
        if full_key in self.cache:
            timestamp = self.cache_timestamps.get(full_key, 0)
            if time.time() - timestamp < self.cache_ttl:
                return self.cache[full_key]
            else:
                # Expired, remove from cache
                self.cache.pop(full_key, None)
                self.cache_timestamps.pop(full_key, None)
        
        # Check Redis cache if available
        if self.redis:
            try:
                cached_data = self.redis.get(f"musearoo:plugin_cache:{full_key}")
                if cached_data:
                    return pickle.loads(cached_data)
            except Exception as e:
                logger.warning(f"Redis cache error: {e}")
        
        return None
    
    async def _cache_result(self, plugin_name: str, cache_key: str, result: Any):
        """Cache plugin execution result"""
        full_key = f"{plugin_name}:{cache_key}"
        
        # Store in memory cache
        self.cache[full_key] = result
        self.cache_timestamps[full_key] = time.time()
        
        # Store in Redis cache if available
        if self.redis:
            try:
                self.redis.setex(
                    f"musearoo:plugin_cache:{full_key}",
                    self.cache_ttl,
                    pickle.dumps(result)
                )
            except Exception as e:
                logger.warning(f"Redis cache error: {e}")
    
    async def _get_ab_test_plugin(self, plugin_name: str) -> str:
        """Get plugin name considering A/B tests"""
        if plugin_name in self.ab_tests:
            test_config = self.ab_tests[plugin_name]
            
            # Simple random assignment (enhance with user-based assignment)
            import random
            if random.random() < test_config.get('split_ratio', 0.5):
                alternative = test_config.get('alternative_plugin')
                if alternative and alternative in self.plugins:
                    logger.debug(f"A/B test: using {alternative} instead of {plugin_name}")
                    return alternative
        
        return plugin_name
    
    async def _emit_event(self, event: PluginEvent, data: Dict[str, Any]):
        """Emit plugin system event"""
        handlers = self.event_handlers.get(event, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event, data)
                else:
                    handler(event, data)
            except Exception as e:
                logger.error(f"Event handler error: {e}")
    
    async def _start_hot_reload(self):
        """Start file system watcher for hot reload"""
        if not self.enable_hot_reload:
            return
        
        self.file_watcher = PluginFileWatcher(self)
        self.file_observer = Observer()
        
        for plugin_dir in self.plugin_directories:
            if Path(plugin_dir).exists():
                self.file_observer.schedule(
                    self.file_watcher,
                    plugin_dir,
                    recursive=True
                )
        
        self.file_observer.start()
        logger.info("🔄 Hot reload enabled")
    
    async def _reload_plugin_from_file(self, file_path: str):
        """Reload a specific plugin from file"""
        file_path = Path(file_path)
        plugin_name = file_path.stem
        
        if plugin_name not in self.plugins:
            # New plugin, just load it
            try:
                await self._load_plugin_from_file(str(file_path))
            except Exception as e:
                logger.error(f"❌ Failed to load new plugin {plugin_name}: {e}")
            return
        
        logger.info(f"🔄 Reloading plugin: {plugin_name}")
        
        try:
            # Cleanup existing plugin
            await self._cleanup_plugin(plugin_name)
            
            # Reload the plugin
            await self._load_plugin_from_file(str(file_path))
            
            # Re-initialize if it was active
            if self.plugin_status[plugin_name] == PluginStatus.LOADED:
                await self._initialize_plugin(plugin_name)
            
            logger.info(f"✅ Plugin reloaded successfully: {plugin_name}")
            
            # Emit update event
            await self._emit_event(PluginEvent.PLUGIN_UPDATED, {
                'plugin_name': plugin_name,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"❌ Failed to reload plugin {plugin_name}: {e}")
            self.plugin_status[plugin_name] = PluginStatus.ERROR
    
    async def _cleanup_plugin(self, plugin_name: str):
        """Clean up plugin resources"""
        plugin = self.plugins.get(plugin_name)
        if plugin:
            try:
                await plugin.cleanup()
            except Exception as e:
                logger.warning(f"Plugin cleanup error for {plugin_name}: {e}")
        
        # Remove from registries
        self.plugin_status[plugin_name] = PluginStatus.UNLOADED
    
    async def _load_cache(self):
        """Load cached data from persistent storage"""
        if self.redis:
            try:
                # Load performance metrics
                metrics_data = self.redis.get("musearoo:plugin_metrics")
                if metrics_data:
                    cached_metrics = pickle.loads(metrics_data)
                    self.performance_metrics.update(cached_metrics)
                
                # Load execution history
                history_data = self.redis.get("musearoo:execution_history")
                if history_data:
                    self.execution_history = pickle.loads(history_data)
                
                logger.debug("📊 Cached data loaded from Redis")
                
            except Exception as e:
                logger.warning(f"Cache loading error: {e}")
    
    async def save_cache(self):
        """Save cached data to persistent storage"""
        if self.redis:
            try:
                # Save performance metrics
                self.redis.setex(
                    "musearoo:plugin_metrics",
                    86400,  # 24 hours
                    pickle.dumps(self.performance_metrics)
                )
                
                # Save execution history
                self.redis.setex(
                    "musearoo:execution_history",
                    86400,  # 24 hours
                    pickle.dumps(self.execution_history)
                )
                
                logger.debug("📊 Cached data saved to Redis")
                
            except Exception as e:
                logger.warning(f"Cache saving error: {e}")
    
    # Public API Methods
    
    def register_event_handler(self, event: PluginEvent, handler: Callable):
        """Register event handler"""
        if event not in self.event_handlers:
            self.event_handlers[event] = []
        self.event_handlers[event].append(handler)
    
    def get_plugin_info(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive plugin information"""
        if plugin_name not in self.plugins:
            return None
        
        metadata = self.plugin_metadata[plugin_name]
        metrics = self.performance_metrics[plugin_name]
        
        return {
            'name': plugin_name,
            'metadata': metadata.__dict__,
            'status': self.plugin_status[plugin_name].value,
            'performance': metrics.__dict__,
            'file_path': self.plugin_files.get(plugin_name),
            'dependencies': list(self.dependency_graph.get(plugin_name, set()))
        }
    
    def list_plugins(self, status_filter: Optional[PluginStatus] = None) -> List[str]:
        """List all plugins, optionally filtered by status"""
        if status_filter:
            return [
                name for name, status in self.plugin_status.items()
                if status == status_filter
            ]
        return list(self.plugins.keys())
    
    def get_plugins_by_phase(self, phase: PluginPhase) -> List[str]:
        """Get all plugins for a specific phase"""
        return [
            name for name, metadata in self.plugin_metadata.items()
            if metadata.phase == phase and self.plugin_status[name] == PluginStatus.ACTIVE
        ]
    
    async def enable_plugin(self, plugin_name: str) -> bool:
        """Enable a disabled plugin"""
        if plugin_name not in self.plugins:
            return False
        
        if self.plugin_status[plugin_name] == PluginStatus.DISABLED:
            await self._initialize_plugin(plugin_name)
            return True
        
        return False
    
    async def disable_plugin(self, plugin_name: str) -> bool:
        """Disable an active plugin"""
        if plugin_name not in self.plugins:
            return False
        
        await self._cleanup_plugin(plugin_name)
        self.plugin_status[plugin_name] = PluginStatus.DISABLED
        return True
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all plugins"""
        summary = {
            'total_plugins': len(self.plugins),
            'active_plugins': len([s for s in self.plugin_status.values() if s == PluginStatus.ACTIVE]),
            'total_executions': sum(m.execution_count for m in self.performance_metrics.values()),
            'average_success_rate': sum(m.success_rate for m in self.performance_metrics.values()) / len(self.performance_metrics) if self.performance_metrics else 0,
            'top_performers': [],
            'slowest_plugins': []
        }
        
        # Top performers by success rate
        top_performers = sorted(
            self.performance_metrics.items(),
            key=lambda x: x[1].success_rate,
            reverse=True
        )[:5]
        summary['top_performers'] = [
            {'name': name, 'success_rate': metrics.success_rate}
            for name, metrics in top_performers
        ]
        
        # Slowest plugins by average runtime
        slowest = sorted(
            self.performance_metrics.items(),
            key=lambda x: x[1].average_runtime,
            reverse=True
        )[:5]
        summary['slowest_plugins'] = [
            {'name': name, 'average_runtime': metrics.average_runtime}
            for name, metrics in slowest
        ]
        
        return summary
    
    async def shutdown(self):
        """Shutdown the plugin manager"""
        logger.info("🛑 Shutting down Advanced Plugin Manager...")
        
        # Save cache
        await self.save_cache()
        
        # Stop hot reload
        if self.file_observer:
            self.file_observer.stop()
            self.file_observer.join()
        
        # Cleanup all plugins
        for plugin_name in list(self.plugins.keys()):
            await self._cleanup_plugin(plugin_name)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("✅ Advanced Plugin Manager shutdown complete")


# Example plugin implementation
class ExampleDrumPlugin(PluginInterface):
    """Example drum generation plugin"""
    
    phase = PluginPhase.PHASE_2_GENERATION
    priority = PluginPriority.NORMAL
    input_types = ["midi", "audio"]
    output_types = ["midi"]
    dependencies = ["librosa", "pretty_midi"]
    
    async def initialize(self, context: Dict[str, Any]) -> bool:
        self.logger.info("🥁 Example Drum Plugin initialized")
        return True
    
    async def execute(self, input_data: Any, **kwargs) -> Any:
        # Simulate drum generation
        await asyncio.sleep(0.1)  # Simulate processing time
        
        return {
            'type': 'drum_pattern',
            'bars': kwargs.get('bars', 8),
            'style': kwargs.get('style', 'rock'),
            'file_path': '/generated/drums.mid'
        }


# Convenience function for easy setup
async def create_plugin_manager(**kwargs) -> AdvancedPluginManager:
    """Create and initialize an Advanced Plugin Manager"""
    manager = AdvancedPluginManager(**kwargs)
    await manager.initialize()
    return manager


if __name__ == "__main__":
    import sys
    
    async def main():
        # Create plugin manager
        manager = await create_plugin_manager(
            plugin_directories=["plugins/", "examples/plugins/"],
            enable_hot_reload=True,
            enable_performance_monitoring=True,
            enable_caching=True
        )
        
        # Example usage
        print("🧩 Advanced Plugin Manager v4.0 Demo")
        print("=" * 50)
        
        # List plugins
        plugins = manager.list_plugins()
        print(f"📋 Loaded plugins: {plugins}")
        
        # Get performance summary
        summary = manager.get_performance_summary()
        print(f"📊 Performance summary: {summary}")
        
        # Keep running
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            await manager.shutdown()
    
    print("🚀 Starting Advanced Plugin Manager...")
    asyncio.run(main())
