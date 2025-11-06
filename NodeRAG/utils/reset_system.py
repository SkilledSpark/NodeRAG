"""
Utility script to reset the NodeRAG system by deleting all indices, graph nodes, and cached files.
This provides a command-line interface for system reset functionality.
"""
import argparse
import os
import sys
import yaml


def reset_system_cli():
    """Command-line interface for system reset"""
    parser = argparse.ArgumentParser(
        description='Reset NodeRAG system by deleting all indices, graph nodes, and cached files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Reset system for a specific data folder
  python -m NodeRAG.utils.reset_system -f /path/to/data

  # Reset with automatic confirmation (use with caution!)
  python -m NodeRAG.utils.reset_system -f /path/to/data --yes

  # Dry run - show what would be deleted without actually deleting
  python -m NodeRAG.utils.reset_system -f /path/to/data --dry-run
        """
    )
    
    parser.add_argument(
        '-f', '--folder_path',
        type=str,
        required=True,
        help='The main folder path containing the NodeRAG data'
    )
    
    parser.add_argument(
        '--yes', '-y',
        action='store_true',
        help='Automatically confirm reset without prompting (use with caution!)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be deleted without actually deleting anything'
    )
    
    args = parser.parse_args()
    
    # Validate folder path
    if not os.path.exists(args.folder_path):
        print(f"Error: Folder path '{args.folder_path}' does not exist")
        sys.exit(1)
    
    # Look for config file
    config_path = os.path.join(args.folder_path, 'Node_config.yaml')
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at '{config_path}'")
        print("Please ensure this is a valid NodeRAG data folder")
        sys.exit(1)
    
    # If dry run, list files that would be deleted
    if args.dry_run:
        print("=" * 80)
        print("DRY RUN - Files and directories that would be deleted:")
        print("=" * 80)
        list_files_to_delete(args.folder_path)
        print("\nNo files were actually deleted (dry run mode)")
        sys.exit(0)
    
    # Confirmation prompt (unless --yes is specified)
    if not args.yes:
        print("=" * 80)
        print("WARNING: This will DELETE all the following:")
        print("  - All cached data files (graphs, embeddings, etc.)")
        print("  - All indices and processed documents")
        print("  - Entity image mappings and extracted images")
        print("  - Document processing state")
        print("=" * 80)
        print(f"\nTarget folder: {args.folder_path}")
        response = input("\nAre you sure you want to reset the system? (yes/no): ")
        
        if response.lower() not in ['yes', 'y']:
            print("Reset cancelled.")
            sys.exit(0)
    
    # Perform reset
    try:
        # Import here to avoid circular imports
        from ..config import NodeConfig
        
        # Load config and perform reset
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        node_config = NodeConfig(config)
        node_config.reset_system(confirm=True)
        
    except Exception as e:
        print(f"\nError during reset: {e}")
        sys.exit(1)


def list_files_to_delete(main_folder: str):
    """List all files and directories that would be deleted during reset"""
    cache_dir = os.path.join(main_folder, 'cache')
    info_dir = os.path.join(main_folder, 'info')
    entity_mappings_dir = os.path.join(main_folder, 'entity_image_mappings')
    extracted_images_dir = os.path.join(main_folder, 'extracted_images')
    
    print("\nCache files:")
    cache_files = [
        'embedding.parquet',
        'text.parquet',
        'documents.parquet',
        'text_decomposition.jsonl',
        'semantic_units.parquet',
        'entities.parquet',
        'relationship.parquet',
        'new_graph.pkl',
        'attributes.parquet',
        'embedding_cache.jsonl',
        'graph.pkl',
        'community_summary.jsonl',
        'high_level_elements.parquet',
        'high_level_elements_titles.parquet',
        'HNSW.bin',
        'hnsw_graph.pkl',
        'id_map.parquet',
        'LLM_error.jsonl',
    ]
    
    for filename in cache_files:
        filepath = os.path.join(cache_dir, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"  ✓ {filename} ({format_size(size)})")
        else:
            print(f"  ✗ {filename} (not found)")
    
    print("\nInfo files:")
    info_files = ['indices.json', 'document_hash.json']
    for filename in info_files:
        filepath = os.path.join(info_dir, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"  ✓ {filename} ({format_size(size)})")
        else:
            print(f"  ✗ {filename} (not found)")
    
    print("\nDirectories:")
    if os.path.exists(entity_mappings_dir):
        size = get_dir_size(entity_mappings_dir)
        print(f"  ✓ entity_image_mappings/ ({format_size(size)})")
    else:
        print(f"  ✗ entity_image_mappings/ (not found)")
    
    if os.path.exists(extracted_images_dir):
        size = get_dir_size(extracted_images_dir)
        print(f"  ✓ extracted_images/ ({format_size(size)})")
    else:
        print(f"  ✗ extracted_images/ (not found)")


def get_dir_size(path: str) -> int:
    """Calculate total size of a directory"""
    total = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total += os.path.getsize(filepath)
    except Exception:
        pass
    return total


def format_size(size: int) -> str:
    """Format byte size to human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"


if __name__ == '__main__':
    reset_system_cli()
