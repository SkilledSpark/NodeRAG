# System Reset Feature Implementation Summary

## Overview
A comprehensive system reset feature has been added to NodeRAG that allows users to completely clear all indices, graph nodes, and cached files, effectively resetting the system to a clean state ready for fresh indexing.

## Files Created/Modified

### 1. Core Implementation
**File:** `NodeRAG/config/Node_config.py`
- **Modified:** Added `reset_system()` method to the `NodeConfig` class
- **Added Import:** `json` module for state file handling
- **Functionality:**
  - Deletes all cache files (graphs, embeddings, processed data)
  - Removes entity image mappings and extracted images
  - Resets indices to initial state
  - Resets system state to "READY"
  - Provides detailed console output with progress
  - Logs reset operation with timestamp

### 2. Command-Line Utility
**File:** `NodeRAG/utils/reset_system.py` (NEW)
- **Purpose:** Standalone CLI tool for system reset
- **Features:**
  - Interactive confirmation prompts
  - `--yes` flag for automated operations
  - `--dry-run` mode to preview deletions
  - Lists files with sizes before deletion
  - Proper error handling and user feedback
  - Validates folder structure and config files

### 3. Module Export
**File:** `NodeRAG/utils/__init__.py`
- **Modified:** Added `reset_system_cli` to exports
- **Purpose:** Makes the CLI function accessible as a module

### 4. Documentation

#### Main Documentation
**File:** `RESET_SYSTEM.md` (NEW)
- Comprehensive guide covering:
  - All usage methods (CLI and API)
  - Complete list of what gets deleted/preserved
  - Safety features
  - Multiple use case examples
  - Troubleshooting guide
  - Warnings and best practices

#### Quick Reference
**File:** `RESET_QUICK_REFERENCE.md` (NEW)
- Concise reference guide with:
  - Command syntax
  - Code snippets
  - Common use cases
  - Troubleshooting table
  - Quick examples

#### README Update
**File:** `README.md`
- **Modified:** Added "System Reset" section
- Links to detailed documentation
- Shows basic CLI usage

### 5. Examples

#### Example Script
**File:** `examples/reset_system_example.py` (NEW)
- 6 different usage examples:
  1. Basic reset
  2. Reset using from_main_folder
  3. Conditional reset based on cache size
  4. Check files before reset
  5. Reset with error handling
  6. Batch reset multiple projects

#### Test Script
**File:** `examples/test_reset_system.py` (NEW)
- Automated test suite:
  - Creates temporary test environment
  - Tests basic reset functionality
  - Verifies safety features (confirmation required)
  - Validates file preservation
  - Cleans up after tests

## Key Features

### 1. Safety Mechanisms
- **Explicit Confirmation Required:** API method requires `confirm=True`
- **Interactive Prompts:** CLI asks for user confirmation by default
- **Dry Run Mode:** Preview deletions without executing them
- **Input Preservation:** Source documents never touched
- **Config Preservation:** Configuration files are preserved
- **Logging:** All operations logged with timestamps

### 2. Flexibility
- **Multiple Access Methods:**
  - Command-line interface
  - Python API
  - Direct method call on NodeConfig instance
  
- **Automation Support:**
  - `--yes` flag for scripts
  - Batch operations support
  - Error handling for robustness

### 3. User Experience
- **Rich Console Output:** Color-coded status messages
- **Progress Tracking:** Shows which files are deleted
- **File Size Information:** Dry run shows sizes
- **Clear Documentation:** Multiple documentation levels
- **Examples Provided:** 6 working examples

## Usage Examples

### Command Line
```bash
# Interactive reset
python -m NodeRAG.utils.reset_system -f /path/to/data

# Automated reset
python -m NodeRAG.utils.reset_system -f /path/to/data --yes

# Preview only
python -m NodeRAG.utils.reset_system -f /path/to/data --dry-run
```

### Python API
```python
from NodeRAG.config import NodeConfig

# Method 1: From main folder
config = NodeConfig.from_main_folder('/path/to/data')
config.reset_system(confirm=True)

# Method 2: From config file
import yaml
with open('/path/to/data/Node_config.yaml', 'r') as f:
    config_data = yaml.safe_load(f)
config = NodeConfig(config_data)
config.reset_system(confirm=True)
```

## What Gets Reset

### Deleted Items
- All cache files (`.parquet`, `.pkl`, `.bin`, `.jsonl`)
  - Embeddings, graphs, indices, mappings
- Entity image mappings directory
- Extracted images directory
- Index counters (reset to 0)
- Document hash mappings
- System state (reset to "READY")

### Preserved Items
- `input/` folder (source documents)
- `images/` folder (original images)
- `Node_config.yaml` (configuration)
- `info/info.log` (with reset marker added)

## Testing

Run the test suite to verify functionality:
```bash
python examples/test_reset_system.py
```

Tests include:
- Basic reset operation
- File deletion verification
- State reset verification
- File preservation verification
- Safety mechanism validation
- Error handling

## Integration

The reset feature integrates seamlessly with existing NodeRAG workflows:

1. **After configuration changes:**
   ```bash
   python -m NodeRAG.utils.reset_system -f ./data --yes
   python -m NodeRAG.build -f ./data
   ```

2. **Fixing corrupted indices:**
   ```bash
   python -m NodeRAG.utils.reset_system -f ./data
   python -m NodeRAG.build -f ./data
   ```

3. **Automated maintenance:**
   ```python
   from NodeRAG.config import NodeConfig
   
   config = NodeConfig.from_main_folder('./data')
   # Check cache size and reset if needed
   if cache_too_large():
       config.reset_system(confirm=True)
   ```

## Error Handling

The implementation includes robust error handling for:
- Missing configuration files
- Permission errors
- File access errors
- Invalid paths
- Interrupted operations

All errors are properly caught and reported with helpful messages.

## Future Enhancements (Possible)

Potential future improvements:
- Selective reset (choose which components to reset)
- Backup before reset option
- Reset history tracking
- GUI integration
- Remote reset capability for multi-machine setups

## Conclusion

The system reset feature provides a safe, flexible, and user-friendly way to completely reset NodeRAG installations. It's well-documented, thoroughly tested, and integrates smoothly with existing workflows.
