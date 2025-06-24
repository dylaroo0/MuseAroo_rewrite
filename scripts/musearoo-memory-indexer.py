#!/usr/bin/env python3
"""
MuseAroo Memory Indexer v1.0
=============================
Creates a persistent knowledge base of your entire project that AI assistants
can reference to maintain context across sessions.

This solves the "AI keeps forgetting" problem by creating:
1. JSON index of all files, classes, functions, and dependencies
2. Markdown documentation for human/AI reading
3. Relationship graphs between modules

Usage:
    python memory_indexer.py /path/to/musearoo/src
    
This will generate:
    - project_manifest.json (machine-readable index)
    - project_manifest.md (human-readable documentation)
    - dependency_graph.json (module relationships)
"""

import ast
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Any, Optional
from datetime import datetime
from collections import defaultdict
import hashlib


class CodeAnalyzer(ast.NodeVisitor):
    """AST visitor to extract code structure."""
    
    def __init__(self):
        self.classes = []
        self.functions = []
        self.imports = []
        self.decorators = []
        self.docstring = None
        
    def visit_Module(self, node):
        """Extract module-level docstring."""
        if ast.get_docstring(node):
            self.docstring = ast.get_docstring(node)
        self.generic_visit(node)
        
    def visit_ClassDef(self, node):
        """Extract class definitions."""
        class_info = {
            "name": node.name,
            "line": node.lineno,
            "docstring": ast.get_docstring(node),
            "methods": [],
            "decorators": [d.id for d in node.decorator_list if isinstance(d, ast.Name)]
        }
        
        # Extract methods
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                class_info["methods"].append({
                    "name": item.name,
                    "line": item.lineno,
                    "is_async": isinstance(item, ast.AsyncFunctionDef)
                })
                
        self.classes.append(class_info)
        self.generic_visit(node)
        
    def visit_FunctionDef(self, node):
        """Extract function definitions."""
        # Skip if inside a class (already captured)
        for parent in ast.walk(node):
            if isinstance(parent, ast.ClassDef):
                return
                
        func_info = {
            "name": node.name,
            "line": node.lineno,
            "docstring": ast.get_docstring(node),
            "is_async": isinstance(node, ast.AsyncFunctionDef),
            "decorators": [d.id for d in node.decorator_list if isinstance(d, ast.Name)]
        }
        self.functions.append(func_info)
        
    def visit_Import(self, node):
        """Extract imports."""
        for alias in node.names:
            self.imports.append({
                "module": alias.name,
                "alias": alias.asname,
                "type": "import"
            })
            
    def visit_ImportFrom(self, node):
        """Extract from imports."""
        module = node.module or ""
        for alias in node.names:
            self.imports.append({
                "module": module,
                "name": alias.name,
                "alias": alias.asname,
                "type": "from"
            })


class MuseArooMemoryIndexer:
    """Creates a comprehensive index of the MuseAroo project."""
    
    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.file_index = {}
        self.dependency_graph = defaultdict(set)
        self.engine_registry = {}
        self.plugin_registry = {}
        self.duplicate_files = defaultdict(list)
        
    def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a single Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Parse AST
            tree = ast.parse(content)
            analyzer = CodeAnalyzer()
            analyzer.visit(tree)
            
            # Calculate file hash for duplicate detection
            file_hash = hashlib.md5(content.encode()).hexdigest()
            
            # Extract file info
            relative_path = file_path.relative_to(self.root_path)
            file_info = {
                "path": str(relative_path),
                "size": len(content),
                "lines": content.count('\n') + 1,
                "hash": file_hash,
                "docstring": analyzer.docstring,
                "classes": analyzer.classes,
                "functions": analyzer.functions,
                "imports": analyzer.imports,
                "is_engine": self._is_engine_file(relative_path, analyzer),
                "is_plugin": self._is_plugin_file(relative_path, analyzer),
                "has_ui_controls": self._has_ui_controls(analyzer),
                "dependencies": self._extract_dependencies(analyzer.imports)
            }
            
            # Detect special file types
            if "drummaroo" in str(relative_path).lower():
                file_info["engine_type"] = "drums"
            elif "bassaroo" in str(relative_path).lower():
                file_info["engine_type"] = "bass"
            elif "melodyroo" in str(relative_path).lower():
                file_info["engine_type"] = "melody"
            elif "harmonyroo" in str(relative_path).lower():
                file_info["engine_type"] = "harmony"
            elif "brainaroo" in str(relative_path).lower():
                file_info["engine_type"] = "analysis"
                
            return file_info
            
        except Exception as e:
            return {
                "path": str(file_path.relative_to(self.root_path)),
                "error": str(e),
                "parseable": False
            }
            
    def _is_engine_file(self, path: Path, analyzer: CodeAnalyzer) -> bool:
        """Detect if file is an engine."""
        path_str = str(path).lower()
        if "engine" in path_str or any(roo in path_str for roo in ["drummaroo", "bassaroo", "melodyroo", "harmonyroo"]):
            return True
            
        # Check for engine base classes
        for cls in analyzer.classes:
            if any(engine in cls["name"].lower() for engine in ["engine", "roo"]):
                return True
                
        return False
        
    def _is_plugin_file(self, path: Path, analyzer: CodeAnalyzer) -> bool:
        """Detect if file is a plugin."""
        path_str = str(path).lower()
        if "plugin" in path_str:
            return True
            
        # Check for plugin decorators
        for cls in analyzer.classes:
            if "plugin" in [d.lower() for d in cls.get("decorators", [])]:
                return True
                
        return False
        
    def _has_ui_controls(self, analyzer: CodeAnalyzer) -> bool:
        """Detect if file has UI control definitions."""
        for cls in analyzer.classes:
            if "control" in cls["name"].lower() or "ui" in cls["name"].lower():
                return True
        return False
        
    def _extract_dependencies(self, imports: List[Dict]) -> List[str]:
        """Extract project-internal dependencies."""
        deps = []
        for imp in imports:
            module = imp.get("module", "")
            if module and not module.startswith((".", "__")):
                # Filter for project modules
                if any(proj in module for proj in ["musearoo", "engines", "context", "utils", "ui"]):
                    deps.append(module)
        return list(set(deps))
        
    def scan_directory(self) -> None:
        """Scan entire directory tree."""
        print(f"🔍 Scanning {self.root_path}...")
        
        python_files = []
        for root, dirs, files in os.walk(self.root_path):
            # Skip hidden and cache directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            for file in files:
                if file.endswith('.py') and not file.startswith('.'):
                    python_files.append(Path(root) / file)
                    
        print(f"📁 Found {len(python_files)} Python files")
        
        # Analyze each file
        for i, file_path in enumerate(python_files):
            print(f"  Analyzing {i+1}/{len(python_files)}: {file_path.name}", end='\r')
            
            file_info = self.analyze_file(file_path)
            self.file_index[str(file_path)] = file_info
            
            # Track duplicates
            if "hash" in file_info:
                self.duplicate_files[file_info["hash"]].append(str(file_path))
                
            # Build dependency graph
            if "dependencies" in file_info:
                for dep in file_info["dependencies"]:
                    self.dependency_graph[str(file_path)].add(dep)
                    
        print("\n✅ Analysis complete!")
        
    def generate_manifest(self) -> None:
        """Generate the manifest files."""
        # Find duplicate groups
        duplicates = {k: v for k, v in self.duplicate_files.items() if len(v) > 1}
        
        # Categorize files
        engines = []
        plugins = []
        ui_files = []
        utils = []
        tests = []
        other = []
        
        for path, info in self.file_index.items():
            if info.get("error"):
                continue
                
            if "test" in path.lower():
                tests.append((path, info))
            elif info.get("is_engine"):
                engines.append((path, info))
            elif info.get("is_plugin"):
                plugins.append((path, info))
            elif info.get("has_ui_controls") or "ui/" in path:
                ui_files.append((path, info))
            elif "utils/" in path:
                utils.append((path, info))
            else:
                other.append((path, info))
                
        # Create JSON manifest
        manifest = {
            "generated": datetime.now().isoformat(),
            "root_path": str(self.root_path),
            "statistics": {
                "total_files": len(self.file_index),
                "total_lines": sum(f.get("lines", 0) for f in self.file_index.values()),
                "engines": len(engines),
                "plugins": len(plugins),
                "ui_components": len(ui_files),
                "duplicate_groups": len(duplicates)
            },
            "files": self.file_index,
            "dependency_graph": {k: list(v) for k, v in self.dependency_graph.items()},
            "duplicates": duplicates,
            "categorized": {
                "engines": [p for p, _ in engines],
                "plugins": [p for p, _ in plugins],
                "ui": [p for p, _ in ui_files],
                "utils": [p for p, _ in utils],
                "tests": [p for p, _ in tests]
            }
        }
        
        # Write JSON
        manifest_path = self.root_path / "project_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
            
        print(f"📄 Generated {manifest_path}")
        
        # Generate Markdown documentation
        self.generate_markdown_report(manifest, engines, plugins, ui_files, duplicates)
        
    def generate_markdown_report(self, manifest: Dict, engines: List, plugins: List, 
                                ui_files: List, duplicates: Dict) -> None:
        """Generate human-readable Markdown report."""
        md_path = self.root_path / "project_manifest.md"
        
        with open(md_path, 'w') as f:
            f.write("# MuseAroo Project Manifest\n\n")
            f.write(f"Generated: {manifest['generated']}\n\n")
            
            # Statistics
            f.write("## 📊 Project Statistics\n\n")
            stats = manifest['statistics']
            f.write(f"- **Total Files:** {stats['total_files']}\n")
            f.write(f"- **Total Lines:** {stats['total_lines']:,}\n")
            f.write(f"- **Engines:** {stats['engines']}\n")
            f.write(f"- **Plugins:** {stats['plugins']}\n")
            f.write(f"- **UI Components:** {stats['ui_components']}\n")
            f.write(f"- **Duplicate File Groups:** {stats['duplicate_groups']}\n\n")
            
            # Engines
            f.write("## 🎛️ Music Engines\n\n")
            for path, info in sorted(engines, key=lambda x: x[0]):
                f.write(f"### {Path(path).name}\n")
                f.write(f"- **Path:** `{path}`\n")
                if info.get("docstring"):
                    f.write(f"- **Description:** {info['docstring'].split('\\n')[0]}\n")
                if info.get("engine_type"):
                    f.write(f"- **Type:** {info['engine_type'].title()}\n")
                f.write(f"- **Classes:** {len(info.get('classes', []))}\n")
                f.write(f"- **Functions:** {len(info.get('functions', []))}\n")
                f.write("\n")
                
            # Duplicate files
            if duplicates:
                f.write("## 🔄 Duplicate Files (Action Required!)\n\n")
                for hash_val, files in duplicates.items():
                    f.write(f"### Duplicate Group (Hash: {hash_val[:8]}...)\n")
                    for file in files:
                        f.write(f"- `{file}`\n")
                    f.write("\n")
                    
            # Dependency hotspots
            f.write("## 🔗 Key Dependencies\n\n")
            dep_counts = defaultdict(int)
            for deps in self.dependency_graph.values():
                for dep in deps:
                    dep_counts[dep] += 1
                    
            for dep, count in sorted(dep_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                f.write(f"- `{dep}` - used by {count} files\n")
                
        print(f"📝 Generated {md_path}")
        
        
def main():
    """CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: python memory_indexer.py /path/to/musearoo/src")
        sys.exit(1)
        
    root_path = sys.argv[1]
    if not os.path.exists(root_path):
        print(f"Error: Path {root_path} does not exist")
        sys.exit(1)
        
    indexer = MuseArooMemoryIndexer(root_path)
    indexer.scan_directory()
    indexer.generate_manifest()
    
    print("\n🎉 Memory index created successfully!")
    print("   - project_manifest.json (for AI agents)")
    print("   - project_manifest.md (for humans)")
    print("\nNext steps:")
    print("1. Review project_manifest.md for duplicate files")
    print("2. Use the JSON manifest in your AI assistant prompts")
    print("3. Run this again after major refactoring")
    

if __name__ == "__main__":
    main()
