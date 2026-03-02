#!/usr/bin/env python
"""
code_modifier.py

Safely modify code based on ChatGPT suggestions.
Includes backup, validation, and rollback functionality.
"""

import ast
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import re
import inspect


class CodeModifier:
    """Safely modify Python code based on suggestions."""
    
    def __init__(self, target_file: Path):
        """
        Initialize code modifier.
        
        Args:
            target_file: Path to file to modify (e.g., rhyme_detector.py)
        """
        self.target_file = Path(target_file)
        self.backup_dir = self.target_file.parent / ".backups"
        self.backup_dir.mkdir(exist_ok=True)
    
    def backup(self) -> Path:
        """Create backup of current file."""
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"{self.target_file.stem}_{timestamp}.py"
        shutil.copy2(self.target_file, backup_path)
        logging.info(f"[BACKUP] Created backup: {backup_path}")
        return backup_path
    
    def validate_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        Validate Python syntax.
        
        Returns:
            (is_valid, error_message)
        """
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, f"Syntax error: {e.msg} at line {e.lineno}"
        except Exception as e:
            return False, f"Parse error: {str(e)}"
    
    def _find_function_in_code(self, func_name: str, code: str) -> Optional[Tuple[int, int]]:
        """Find function definition in code using AST."""
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == func_name:
                    # Get line numbers
                    start_line = node.lineno - 1  # Convert to 0-indexed
                    # Find end of function (next function/class or end of file)
                    lines = code.split('\n')
                    end_line = len(lines)
                    
                    # Find next top-level definition
                    for sibling in ast.walk(tree):
                        if isinstance(sibling, (ast.FunctionDef, ast.ClassDef)) and sibling != node:
                            if sibling.lineno > node.lineno:
                                end_line = min(end_line, sibling.lineno - 1)
                                break
                    
                    return (start_line, end_line)
        except Exception:
            pass
        return None
    
    def _find_class_in_code(self, class_name: str, code: str) -> Optional[Tuple[int, int]]:
        """Find class definition in code using AST."""
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    start_line = node.lineno - 1
                    lines = code.split('\n')
                    end_line = len(lines)
                    
                    # Find next top-level definition
                    for sibling in ast.walk(tree):
                        if isinstance(sibling, (ast.FunctionDef, ast.ClassDef)) and sibling != node:
                            if sibling.lineno > node.lineno:
                                end_line = min(end_line, sibling.lineno - 1)
                                break
                    
                    return (start_line, end_line)
        except Exception:
            pass
        return None
    
    def _extract_function_name_from_code(self, code: str) -> Optional[str]:
        """Extract function or class name from code snippet."""
        # Try to find def or class
        match = re.search(r"(?:def|class)\s+(\w+)", code)
        if match:
            return match.group(1)
        return None
    
    def apply_suggestion(
        self,
        suggestion: Dict,
        current_code: str,
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Apply a code modification suggestion with improved location finding.
        
        Args:
            suggestion: Dict with 'location', 'code', 'fix' fields
            current_code: Current file contents
        
        Returns:
            (success, modified_code, error_message)
        """
        location = suggestion.get("location", "").strip()
        new_code = suggestion.get("code", "").strip()
        fix_description = suggestion.get("fix", "").lower()
        
        if not new_code:
            return False, current_code, "No code provided in suggestion"
        
        # Clean up new_code (remove markdown code blocks if present)
        if new_code.startswith("```"):
            lines = new_code.split('\n')
            # Remove first and last lines if they're code block markers
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            new_code = '\n'.join(lines)
        
        modified_code = current_code
        lines = current_code.split('\n')
        
        # Strategy 1: Try to extract name from new_code and find it
        extracted_name = self._extract_function_name_from_code(new_code)
        if extracted_name:
            # Try function first
            func_pos = self._find_function_in_code(extracted_name, current_code)
            if func_pos:
                start_line, end_line = func_pos
                new_lines = lines[:start_line] + new_code.split('\n') + lines[end_line:]
                modified_code = '\n'.join(new_lines)
                logging.info(f"[APPLY] Replaced function {extracted_name} using AST (lines {start_line+1}-{end_line+1})")
            else:
                # Try class
                class_pos = self._find_class_in_code(extracted_name, current_code)
                if class_pos:
                    start_line, end_line = class_pos
                    new_lines = lines[:start_line] + new_code.split('\n') + lines[end_line:]
                    modified_code = '\n'.join(new_lines)
                    logging.info(f"[APPLY] Replaced class {extracted_name} using AST (lines {start_line+1}-{end_line+1})")
        
        # Strategy 2: Use location hint from suggestion
        if modified_code == current_code and location:
            # Try different location formats
            # Format: "function_name" or "class_name" or "function_name:lines" or "lines:start-end"
            
            # Try line range format
            if ":" in location and "-" in location:
                try:
                    # Format: "lines:10-20"
                    parts = location.split(":")
                    if len(parts) == 2 and "-" in parts[1]:
                        start, end = map(int, parts[1].split("-"))
                        if 1 <= start <= len(lines) and 1 <= end <= len(lines):
                            new_lines = lines[:start-1] + new_code.split('\n') + lines[end:]
                            modified_code = '\n'.join(new_lines)
                            logging.info(f"[APPLY] Replaced lines {start}-{end} using location hint")
                except ValueError:
                    pass
            
            # Try function/class name
            if modified_code == current_code:
                func_pos = self._find_function_in_code(location, current_code)
                if func_pos:
                    start_line, end_line = func_pos
                    new_lines = lines[:start_line] + new_code.split('\n') + lines[end_line:]
                    modified_code = '\n'.join(new_lines)
                    logging.info(f"[APPLY] Replaced function {location} using location hint")
                else:
                    class_pos = self._find_class_in_code(location, current_code)
                    if class_pos:
                        start_line, end_line = class_pos
                        new_lines = lines[:start_line] + new_code.split('\n') + lines[end_line:]
                        modified_code = '\n'.join(new_lines)
                        logging.info(f"[APPLY] Replaced class {location} using location hint")
            
            # Try regex pattern matching
            if modified_code == current_code:
                # Look for function definition with this name
                func_pattern = rf"(def\s+{re.escape(location)}\s*\([^)]*\):.*?)(?=\n\s*(?:def|class|@|\Z))"
                func_match = re.search(func_pattern, current_code, re.DOTALL | re.MULTILINE)
                if func_match:
                    modified_code = current_code[:func_match.start()] + new_code + "\n\n" + current_code[func_match.end():]
                    logging.info(f"[APPLY] Replaced function {location} using regex")
                else:
                    class_pattern = rf"(class\s+{re.escape(location)}\s*:.*?)(?=\n\s*(?:class|def|\Z))"
                    class_match = re.search(class_pattern, current_code, re.DOTALL | re.MULTILINE)
                    if class_match:
                        modified_code = current_code[:class_match.start()] + new_code + "\n\n" + current_code[class_match.end():]
                        logging.info(f"[APPLY] Replaced class {location} using regex")
        
        # Strategy 3: Insert new code (for "add" or "insert" operations)
        if ("add" in fix_description or "insert" in fix_description) and modified_code == current_code:
            # Try to find insertion point
            if location:
                # Try to find location and insert after it
                func_pos = self._find_function_in_code(location, current_code)
                if func_pos:
                    # Insert after the function
                    start_line, end_line = func_pos
                    new_lines = lines[:end_line] + [""] + new_code.split('\n') + lines[end_line:]
                    modified_code = '\n'.join(new_lines)
                    logging.info(f"[APPLY] Inserted code after function {location}")
                else:
                    # Insert at end of file
                    modified_code = current_code + "\n\n" + new_code
                    logging.info(f"[APPLY] Added code at end of file")
            else:
                # Insert at end of file
                modified_code = current_code + "\n\n" + new_code
                logging.info(f"[APPLY] Added code at end of file")
        
        # Strategy 4: If still no change, try to find partial matches
        if modified_code == current_code and extracted_name:
            # Try fuzzy matching - look for similar function names
            # This is a fallback for when exact match fails
            for line_num, line in enumerate(lines):
                if f"def {extracted_name}" in line or f"class {extracted_name}" in line:
                    # Found a potential match, try to replace from here
                    # Find the end of the function/class
                    indent_level = len(line) - len(line.lstrip())
                    end_line = line_num + 1
                    for i in range(line_num + 1, len(lines)):
                        if lines[i].strip() and not lines[i].startswith(' ' * (indent_level + 1)) and not lines[i].startswith('\t'):
                            end_line = i
                            break
                    else:
                        end_line = len(lines)
                    
                    new_lines = lines[:line_num] + new_code.split('\n') + lines[end_line:]
                    modified_code = '\n'.join(new_lines)
                    logging.info(f"[APPLY] Replaced code near line {line_num+1} using fuzzy match")
                    break
        
        # Validate modified code
        if modified_code != current_code:
            is_valid, error = self.validate_syntax(modified_code)
            if not is_valid:
                logging.warning(f"Syntax validation failed: {error}")
                return False, current_code, error
            return True, modified_code, None
        
        # If we got here, we couldn't apply the change
        return False, current_code, f"Could not find location to modify: {location or 'unknown'}"
    
    def apply_suggestions(
        self,
        suggestions: List[Dict],
        priority_filter: Optional[str] = None,
    ) -> Tuple[bool, str, List[str]]:
        """
        Apply multiple suggestions.
        
        Args:
            suggestions: List of suggestion dicts
            priority_filter: Only apply "high" priority suggestions
        
        Returns:
            (success, modified_code, list_of_errors)
        """
        # Read current code
        current_code = self.target_file.read_text(encoding="utf-8")
        
        # Filter by priority if requested
        if priority_filter:
            suggestions = [s for s in suggestions if s.get("priority", "").lower() == priority_filter.lower()]
        
        # Sort by priority (high -> medium -> low)
        priority_order = {"high": 0, "medium": 1, "low": 2}
        suggestions.sort(key=lambda s: priority_order.get(s.get("priority", "low").lower(), 2))
        
        # Create backup
        backup_path = self.backup()
        
        modified_code = current_code
        errors = []
        applied = []
        
        for suggestion in suggestions:
            success, new_code, error = self.apply_suggestion(suggestion, modified_code)
            if success:
                modified_code = new_code
                applied.append(suggestion.get("issue", "unknown"))
                logging.info(f"[APPLY] Applied: {suggestion.get('issue', 'unknown')}")
            else:
                errors.append(f"{suggestion.get('issue', 'unknown')}: {error}")
                logging.warning(f"[SKIP] Failed to apply: {suggestion.get('issue', 'unknown')} - {error}")
        
        if applied:
            # Write modified code
            self.target_file.write_text(modified_code, encoding="utf-8")
            logging.info(f"[SUCCESS] Applied {len(applied)} suggestions. Backup: {backup_path}")
            return True, modified_code, errors
        else:
            logging.warning(f"[SKIP] No suggestions applied. Restoring from backup.")
            self.restore(backup_path)
            return False, current_code, errors
    
    def restore(self, backup_path: Path) -> None:
        """Restore file from backup."""
        shutil.copy2(backup_path, self.target_file)
        logging.info(f"[RESTORE] Restored from: {backup_path}")
