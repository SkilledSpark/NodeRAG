# System Reset Feature

This document describes the system reset feature that allows you to completely clear all indices, graph nodes, and cached files from your NodeRAG installation.

## Overview

The reset feature provides a safe way to:
- Delete all cached data files (graphs, embeddings, processed documents)
- Clear all indices and counters
- Remove entity image mappings and extracted images
- Reset the system state to initial configuration
- Prepare the system for fresh indexing

## Usage Methods

### Method 1: Command Line Interface (Recommended)

The easiest way to reset the system is using the command-line utility:

```bash
# Basic reset with interactive confirmation
python -m NodeRAG.utils.reset_system -f /path/to/data

# Reset with automatic confirmation (no prompt)
python -m NodeRAG.utils.reset_system -f /path/to/data --yes

# Dry run - see what would be deleted without actually deleting
python -m NodeRAG.utils.reset_system -f /path/to/data --dry-run
```

#### Command Line Options

- `-f, --folder_path`: (Required) Path to your NodeRAG data folder
- `--yes, -y`: Skip confirmation prompt and proceed directly
- `--dry-run`: Show what would be deleted without actually deleting anything

### Method 2: Python API

You can also reset the system programmatically:

```python
import yaml
from NodeRAG.config import NodeConfig

# Load your configuration
config_path = '/path/to/data/Node_config.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Create NodeConfig instance
node_config = NodeConfig(config)

# Perform reset (requires explicit confirmation)
node_config.reset_system(confirm=True)
```

### Method 3: Using from_main_folder

```python
from NodeRAG.config import NodeConfig

# Create config from main folder
node_config = NodeConfig.from_main_folder('/path/to/data')

# Reset the system
node_config.reset_system(confirm=True)
```

## What Gets Deleted

The reset operation removes the following:

### Cache Files (`data/cache/`)
- `embedding.parquet` - Embedding vectors
- `text.parquet` - Processed text chunks
- `documents.parquet` - Document metadata
- `text_decomposition.jsonl` - Text decomposition results
- `semantic_units.parquet` - Semantic units
- `entities.parquet` - Extracted entities
- `relationship.parquet` - Entity relationships
- `new_graph.pkl` / `graph.pkl` - Graph structures
- `attributes.parquet` - Entity attributes
- `embedding_cache.jsonl` - Embedding cache
- `community_summary.jsonl` - Community summaries
- `high_level_elements.parquet` - High-level elements
- `high_level_elements_titles.parquet` - Element titles
- `HNSW.bin` - HNSW index
- `hnsw_graph.pkl` - HNSW graph structure
- `id_map.parquet` - ID mappings
- `LLM_error.jsonl` - LLM error logs

### Info Files (`data/info/`)
- `indices.json` - Index counters (reset to initial state)
- `document_hash.json` - Document hash mappings
- `state.json` - System state (reset to "READY")

### Image Data
- `entity_image_mappings/` - Entity to image mapping files
- `extracted_images/` - Extracted images from documents

### What Does NOT Get Deleted

The following are preserved:
- `input/` - Your source documents (never touched)
- `images/` - Original uploaded images
- `Node_config.yaml` - Configuration file
- `info/info.log` - Log file (a reset marker is added, but history is kept)

## Safety Features

1. **Confirmation Required**: By default, the API method requires `confirm=True` to prevent accidental resets
2. **Interactive Prompt**: The CLI asks for confirmation unless `--yes` is specified
3. **Dry Run Mode**: Use `--dry-run` to preview what would be deleted
4. **Input Preservation**: Source documents in the `input/` folder are never deleted
5. **Logging**: Reset operations are logged with timestamp in `info.log`

## Use Cases

### Starting Fresh After Testing
```bash
# After experimenting with different configurations
python -m NodeRAG.utils.reset_system -f ./data --yes
# Then rebuild with new settings
python -m NodeRAG.build -f ./data
```

### Clearing Corrupted Data
```bash
# If the system is in an inconsistent state
python -m NodeRAG.utils.reset_system -f ./data
# Rebuild from scratch
python -m NodeRAG.build -f ./data
```

### Checking Space Usage
```bash
# See what's consuming space before deciding to reset
python -m NodeRAG.utils.reset_system -f ./data --dry-run
```

### Batch Reset for Multiple Projects
```python
from NodeRAG.config import NodeConfig

projects = ['/path/to/project1', '/path/to/project2', '/path/to/project3']

for project_path in projects:
    config = NodeConfig.from_main_folder(project_path)
    print(f"Resetting {project_path}...")
    config.reset_system(confirm=True)
    print(f"✓ {project_path} reset complete")
```

## After Reset

After resetting the system:

1. **System State**: The system is in "READY" state
2. **Indices**: All counters are reset to 0
3. **Next Steps**: Run the build process to reindex your documents:
   ```bash
   python -m NodeRAG.build -f /path/to/data
   ```

## Troubleshooting

### Permission Errors
If you get permission errors:
- Ensure no other processes are using the files
- Close any open notebooks or applications accessing the data
- Run with appropriate permissions (admin/sudo if needed)

### File Not Found Errors
These are normal - the reset will skip files that don't exist and continue with others.

### Partial Reset
If the reset is interrupted:
- You can safely run it again - it will skip already deleted files
- The system may be in an inconsistent state until reset completes
- Consider using `--dry-run` first to check the current state

## Examples

### Example 1: Safe Reset with Preview
```bash
# First, see what will be deleted
python -m NodeRAG.utils.reset_system -f ./data --dry-run

# Review the output, then proceed
python -m NodeRAG.utils.reset_system -f ./data
```

### Example 2: Scripted Reset
```bash
# For use in scripts/automation
python -m NodeRAG.utils.reset_system -f ./data --yes
```

### Example 3: Conditional Reset in Python
```python
from NodeRAG.config import NodeConfig
import os

config = NodeConfig.from_main_folder('./data')

# Check if we need to reset based on some condition
cache_size = sum(
    os.path.getsize(os.path.join(dirpath, filename))
    for dirpath, _, filenames in os.walk(config.cache)
    for filename in filenames
)

if cache_size > 10 * 1024 * 1024 * 1024:  # > 10GB
    print(f"Cache too large ({cache_size / 1e9:.2f} GB), resetting...")
    config.reset_system(confirm=True)
```

## Warning

⚠️ **This operation is irreversible!** Once files are deleted, they cannot be recovered. Always ensure:
- You have backups of important data
- Your source documents are safe in the `input/` folder
- You understand what will be deleted
- You're ready to rebuild the indices from scratch

## Support

For issues or questions about the reset feature:
1. Check the `info/info.log` file for error messages
2. Ensure your config file is valid
3. Verify folder paths are correct
4. Review this documentation for proper usage
