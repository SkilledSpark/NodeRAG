"""
Utility to check which named entities are linked to images
"""
import os
import json
import sys
from pathlib import Path
from typing import Dict, List

def check_entity_images(mappings_dir: str = None) -> Dict:
    """
    Check which entities are linked to images
    
    Args:
        mappings_dir: Path to entity_image_mappings directory
        
    Returns:
        Dictionary with statistics and entity information
    """
    if mappings_dir is None:
        mappings_dir = 'data'
    
    if not os.path.exists(mappings_dir):
        print(f"❌ Mappings directory not found: {mappings_dir}")
        print(f"\nPlease run the PDF processor first:")
        print(f"   python -m NodeRAG.utils.process_pdfs")
        return {}
    
    # Find all mapping files
    mapping_files = list(Path(mappings_dir).glob('*_entity_image_map.json'))
    
    if not mapping_files:
        print(f"❌ No mapping files found in: {mappings_dir}")
        print(f"\nPlease run the PDF processor first:")
        print(f"   python -m NodeRAG.utils.process_pdfs")
        return {}
    
    print("=" * 70)
    print("ENTITY-IMAGE MAPPING REPORT")
    print("=" * 70)
    print()
    
    # Process each mapping file
    all_entities = {}
    total_entities = 0
    entities_with_images = 0
    total_links = 0
    entity_types = {}
    
    for mapping_file in mapping_files:
        print(f"📄 File: {mapping_file.name}")
        print("-" * 70)
        
        with open(mapping_file, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)
        
        file_entities = len(mapping_data)
        file_with_images = 0
        file_links = 0
        
        for entity_key, entity_data in mapping_data.items():
            total_entities += 1
            entity_type = entity_data.get('label', 'UNKNOWN')
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
            
            if entity_data.get('images'):
                entities_with_images += 1
                file_with_images += 1
                file_links += len(entity_data['images'])
                total_links += len(entity_data['images'])
                
                # Store for detailed view
                all_entities[entity_data['name']] = {
                    'type': entity_type,
                    'images': entity_data['images'],
                    'pages': entity_data.get('pages', []),
                    'source': mapping_file.stem
                }
        
        print(f"  Total entities: {file_entities}")
        print(f"  Entities with images: {file_with_images}")
        print(f"  Total image links: {file_links}")
        print()
    
    # Overall summary
    print("=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    print(f"Total mapping files: {len(mapping_files)}")
    print(f"Total entities found: {total_entities}")
    print(f"Entities with images: {entities_with_images} ({entities_with_images/total_entities*100:.1f}%)" if total_entities > 0 else "Entities with images: 0")
    print(f"Total entity-image links: {total_links}")
    print()
    
    # Entity types breakdown
    if entity_types:
        print("Entity Types:")
        for entity_type, count in sorted(entity_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {entity_type}: {count}")
        print()
    
    return {
        'all_entities': all_entities,
        'total_entities': total_entities,
        'entities_with_images': entities_with_images,
        'total_links': total_links,
        'entity_types': entity_types
    }


def show_entities_with_images(mappings_dir: str = None, limit: int = 20):
    """
    Show detailed list of entities that have linked images
    
    Args:
        mappings_dir: Path to entity_image_mappings directory
        limit: Maximum number of entities to display
    """
    stats = check_entity_images(mappings_dir)
    
    if not stats:
        return
    
    all_entities = stats.get('all_entities', {})
    
    if not all_entities:
        print("❌ No entities with images found.")
        return
    
    print("=" * 70)
    print(f"ENTITIES WITH IMAGES (Showing {min(limit, len(all_entities))} of {len(all_entities)})")
    print("=" * 70)
    print()
    
    for idx, (entity_name, entity_info) in enumerate(list(all_entities.items())[:limit], 1):
        print(f"{idx}. {entity_name}")
        print(f"   Type: {entity_info['type']}")
        print(f"   Pages: {entity_info['pages']}")
        print(f"   Source: {entity_info['source']}")
        print(f"   Images ({len(entity_info['images'])}):")
        for img_path in entity_info['images']:
            print(f"      • {Path(img_path).name}")
        print()


def find_entity_images(entity_name: str, mappings_dir: str = None) -> List[str]:
    """
    Find images for a specific entity
    
    Args:
        entity_name: Name of the entity to search for
        mappings_dir: Path to entity_image_mappings directory
        
    Returns:
        List of image paths
    """
    if mappings_dir is None:
        mappings_dir = 'data'
    if not os.path.exists(mappings_dir):
        print(f"❌ Mappings directory not found: {mappings_dir}")
        return []
    
    mapping_files = list(Path(mappings_dir).glob('*_entity_image_map.json'))
    
    entity_lower = entity_name.lower()
    found_images = []
    
    for mapping_file in mapping_files:
        with open(mapping_file, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)
        
        if entity_lower in mapping_data:
            entity_data = mapping_data[entity_lower]
            images = entity_data.get('images', [])
            if images:
                print(f"✅ Found entity '{entity_data['name']}' in {mapping_file.name}")
                print(f"   Type: {entity_data.get('label', 'UNKNOWN')}")
                print(f"   Pages: {entity_data.get('pages', [])}")
                print(f"   Images ({len(images)}):")
                for img in images:
                    print(f"      • {img}")
                    found_images.append(img)
                print()
    
    if not found_images:
        print(f"❌ No images found for entity: {entity_name}")
        print(f"\nTip: Entity names are case-sensitive in the search.")
        print(f"     Try checking the mapping files to see exact entity names.")
    
    return found_images


def search_entities_by_type(entity_type: str, mappings_dir: str = None, limit: int = 10):
    """
    Find entities of a specific type
    
    Args:
        entity_type: Entity type (PERSON, ORG, GPE, etc.)
        mappings_dir: Path to entity_image_mappings directory
        limit: Maximum number to show
    """
    if mappings_dir is None:
        mappings_dir = r'data'
    if not os.path.exists(mappings_dir):
        print(f"❌ Mappings directory not found: {mappings_dir}")
        return
    
    mapping_files = list(Path(mappings_dir).glob('*_entity_image_map.json'))
    
    matching_entities = []
    entity_type_upper = entity_type.upper()
    
    for mapping_file in mapping_files:
        with open(mapping_file, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)
        
        for entity_key, entity_data in mapping_data.items():
            if entity_data.get('label', '').upper() == entity_type_upper:
                matching_entities.append({
                    'name': entity_data['name'],
                    'has_images': bool(entity_data.get('images')),
                    'image_count': len(entity_data.get('images', [])),
                    'pages': entity_data.get('pages', []),
                    'source': mapping_file.stem
                })
    
    if not matching_entities:
        print(f"❌ No entities of type '{entity_type}' found.")
        return
    
    print("=" * 70)
    print(f"ENTITIES OF TYPE: {entity_type}")
    print("=" * 70)
    print(f"Found {len(matching_entities)} entities")
    print()
    
    # Show entities with images first
    with_images = [e for e in matching_entities if e['has_images']]
    without_images = [e for e in matching_entities if not e['has_images']]
    
    if with_images:
        print(f"Entities with images ({len(with_images)}):")
        for idx, entity in enumerate(with_images[:limit], 1):
            print(f"  {idx}. {entity['name']} - {entity['image_count']} image(s) on pages {entity['pages']}")
        print()
    
    if without_images and limit > len(with_images):
        print(f"Entities without images ({len(without_images)}):")
        remaining = limit - len(with_images)
        for idx, entity in enumerate(without_images[:remaining], 1):
            print(f"  {idx}. {entity['name']} on pages {entity['pages']}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Check entity-image mappings')
    parser.add_argument('--dir', help='Path to entity_image_mappings directory', default=None)
    parser.add_argument('--find', help='Find images for a specific entity', default=None)
    parser.add_argument('--type', help='Show entities of a specific type (PERSON, ORG, GPE, etc.)', default=None)
    parser.add_argument('--limit', type=int, help='Limit results', default=20)
    parser.add_argument('--summary', action='store_true', help='Show summary only')
    
    args = parser.parse_args()
    
    if args.find:
        find_entity_images(args.find, args.dir)
    elif args.type:
        search_entities_by_type(args.type, args.dir, args.limit)
    elif args.summary:
        check_entity_images(args.dir)
    else:
        show_entities_with_images(args.dir, args.limit)
