# Changelog - System Reset Feature

## [Added] System Reset Feature - 2025-11-06

### New Features

#### 1. System Reset Method
- Added `reset_system(confirm: bool = False)` method to `NodeConfig` class
- Completely resets the NodeRAG system by deleting all processed data
- Requires explicit confirmation to prevent accidental data loss
- Preserves source documents and configuration files

#### 2. Command-Line Interface
- New CLI tool: `python -m NodeRAG.utils.reset_system`
- Interactive confirmation prompts for safety
- `--yes` flag for automated/scripted operations
- `--dry-run` mode to preview deletions without executing

#### 3. Comprehensive Documentation
- **RESET_SYSTEM.md**: Full documentation with examples and use cases
- **RESET_QUICK_REFERENCE.md**: Quick reference guide for common operations
- Updated **README.md** with reset feature section

#### 4. Example Scripts
- **examples/reset_system_example.py**: 6 different usage patterns
- **examples/test_reset_system.py**: Automated test suite

### Technical Details

#### Files Modified
- `NodeRAG/config/Node_config.py`
  - Added `reset_system()` method
  - Added `json` import for state file handling
  
- `NodeRAG/utils/__init__.py`
  - Exported `reset_system_cli` function

#### Files Created
- `NodeRAG/utils/reset_system.py` - CLI utility
- `RESET_SYSTEM.md` - Full documentation
- `RESET_QUICK_REFERENCE.md` - Quick reference
- `IMPLEMENTATION_SUMMARY.md` - Implementation details
- `examples/reset_system_example.py` - Usage examples
- `examples/test_reset_system.py` - Test suite

### What Gets Reset

#### Deleted
- All cache files (`.parquet`, `.pkl`, `.bin`, `.jsonl`)
- Entity image mappings directory
- Extracted images directory  
- Index counters (reset to 0)
- Document hash mappings
- System state (reset to "READY")

#### Preserved
- `input/` directory (source documents)
- `images/` directory (original images)
- `Node_config.yaml` (configuration)
- `info/info.log` (with reset marker added)

### Safety Features
1. Explicit confirmation required (`confirm=True` parameter)
2. Interactive CLI prompts by default
3. Dry-run mode available for previewing
4. Source documents never touched
5. All operations logged with timestamps
6. Detailed progress output

### Usage Examples

#### Command Line
```bash
# Interactive reset
python -m NodeRAG.utils.reset_system -f /path/to/data

# Automated reset
python -m NodeRAG.utils.reset_system -f /path/to/data --yes

# Preview only
python -m NodeRAG.utils.reset_system -f /path/to/data --dry-run
```

#### Python API
```python
from NodeRAG.config import NodeConfig

# Reset using main folder
config = NodeConfig.from_main_folder('/path/to/data')
config.reset_system(confirm=True)
```

### Use Cases
- Starting fresh after configuration changes
- Fixing corrupted indices or inconsistent state
- Clearing test data during development
- Reclaiming disk space
- Preparing for clean rebuild

### Testing
- Automated test suite included
- Tests verify correct file deletion
- Tests verify file preservation
- Tests verify safety mechanisms
- Run with: `python examples/test_reset_system.py`

### Documentation
- Full guide: `RESET_SYSTEM.md`
- Quick reference: `RESET_QUICK_REFERENCE.md`
- Implementation details: `IMPLEMENTATION_SUMMARY.md`
- Usage examples: `examples/reset_system_example.py`
- Test suite: `examples/test_reset_system.py`

### Breaking Changes
None - This is a new feature with no impact on existing functionality

### Dependencies
No new dependencies added - uses only standard library and existing NodeRAG components

### Notes
- This feature is designed for development and maintenance purposes
- Always ensure you have backups before resetting production systems
- The reset operation is irreversible
- Source documents in `input/` folder are never affected
