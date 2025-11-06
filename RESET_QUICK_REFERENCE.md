# System Reset - Quick Reference

## Command Line Usage

### Basic Commands

```bash
# Interactive reset (with confirmation prompt)
python -m NodeRAG.utils.reset_system -f /path/to/data

# Reset without prompt (auto-confirm)
python -m NodeRAG.utils.reset_system -f /path/to/data --yes

# Dry run (preview only, no deletion)
python -m NodeRAG.utils.reset_system -f /path/to/data --dry-run
```

## Python API Usage

### Method 1: Basic Reset
```python
from NodeRAG.config import NodeConfig

config = NodeConfig.from_main_folder('/path/to/data')
config.reset_system(confirm=True)
```

### Method 2: With Config File
```python
import yaml
from NodeRAG.config import NodeConfig

with open('/path/to/data/Node_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

node_config = NodeConfig(config)
node_config.reset_system(confirm=True)
```

## What Gets Deleted

✅ **Deleted:**
- All cache files (graphs, embeddings, processed data)
- Index counters (reset to 0)
- Entity image mappings
- Extracted images
- Document hashes
- System state (reset to "READY")

❌ **Preserved:**
- `input/` folder (your source documents)
- `images/` folder (original images)
- `Node_config.yaml` (configuration)
- `info/info.log` (log history with reset marker)

## Common Use Cases

### Clean Start After Configuration Changes
```bash
# Modify your Node_config.yaml, then:
python -m NodeRAG.utils.reset_system -f ./data --yes
python -m NodeRAG.build -f ./data
```

### Fix Corrupted Index
```bash
python -m NodeRAG.utils.reset_system -f ./data
python -m NodeRAG.build -f ./data
```

### Check Space Usage Before Reset
```bash
python -m NodeRAG.utils.reset_system -f ./data --dry-run
```

### Automated Cleanup Script
```python
from NodeRAG.config import NodeConfig

# Reset multiple projects
for path in ['/data/project1', '/data/project2']:
    config = NodeConfig.from_main_folder(path)
    config.reset_system(confirm=True)
```

## Safety Features

1. **Requires explicit confirmation** - Pass `confirm=True` in API or use `--yes` flag
2. **Interactive prompts** - CLI asks for confirmation by default
3. **Dry run mode** - Preview deletions without executing them
4. **Input preservation** - Source documents are never touched
5. **Logged operations** - All resets are recorded in info.log

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Permission denied | Close applications using the files, run with appropriate permissions |
| Config file not found | Ensure you're pointing to a valid NodeRAG data folder |
| Partial reset | Run reset again - it will skip already deleted files |
| Want to undo | Reset is irreversible - always backup important data first |

## After Reset

1. System state is reset to "READY"
2. All indices are at 0
3. Cache is empty
4. Ready to rebuild with: `python -m NodeRAG.build -f /path/to/data`

## Examples

See `examples/reset_system_example.py` for detailed examples including:
- Basic reset
- Conditional reset based on cache size
- Error handling
- Batch operations
- Pre-reset validation

## Full Documentation

For complete documentation, see [RESET_SYSTEM.md](./RESET_SYSTEM.md)
