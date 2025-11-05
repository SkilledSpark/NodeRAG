"""
Script to process PDF files and generate entity-image mappings
This should be run after documents are uploaded to the input folder.

Notes:
- The script can be executed from any working directory; it will resolve paths
    relative to the project root automatically.
- Default input folder is the project_root/data directory and PDFs are searched
    recursively within it (so project_root/data/input is also supported).
"""
import os
import sys
from pathlib import Path
from typing import Optional

# Ensure we import from the project root (parent of the 'NodeRAG' package)
_THIS_FILE = Path(__file__).resolve()
_PACKAGE_ROOT = _THIS_FILE.parent.parent  # .../NodeRAG
_REPO_ROOT = _PACKAGE_ROOT.parent         # repo root that contains the 'NodeRAG' package
if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

def _resolve_path(path_like: str) -> str:
        """Resolve a potentially relative path against the project root."""
        p = Path(path_like)
        if not p.is_absolute():
                p = _REPO_ROOT / p
        return str(p.resolve())


def process_pdfs_in_folder(input_folder: str, output_base_dir: Optional[str] = None):
    """
    Process all PDF files in a folder and generate entity-image mappings
    
    Args:
        input_folder: Path to folder containing PDF files
        output_base_dir: Base directory for outputs (defaults to parent of input_folder)
    """
    # Local import after sys.path is adjusted at module import-time
    from NodeRAG.utils.pdf_extractor import PDFExtractor
    from NodeRAG.utils.entity_image_linker import EntityImageLinker
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist")
        return
    
    # Set up output directories
    if output_base_dir is None:
        # If the provided input folder is an 'input' subfolder, write outputs next to it
        if os.path.basename(input_folder.rstrip('/\\')).lower() == 'input':
            output_base_dir = os.path.dirname(input_folder)
        else:
            output_base_dir = input_folder
    
    images_output = os.path.join(output_base_dir, 'extracted_images')
    mappings_output = os.path.join(output_base_dir, 'entity_image_mappings')
    
    os.makedirs(images_output, exist_ok=True)
    os.makedirs(mappings_output, exist_ok=True)
    
    # Find all PDF files
    pdf_files = list(Path(input_folder).glob('**/*.pdf'))
    
    if not pdf_files:
        print(f"No PDF files found in '{input_folder}'")
        return
    
    print(f"Found {len(pdf_files)} PDF file(s) to process")
    print(f"Images will be saved to: {images_output}")
    print(f"Mappings will be saved to: {mappings_output}")
    print()
    
    # Process each PDF
    results = []
    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}...")
        try:
            # Create subdirectory for this PDF's images
            pdf_images_dir = os.path.join(images_output, pdf_path.stem)
            os.makedirs(pdf_images_dir, exist_ok=True)
            
            # Extract PDF content
            extractor = PDFExtractor(pdf_images_dir)
            extracted_data = extractor.extract(str(pdf_path))
            
            print(f"  - Extracted {extracted_data['metadata']['total_pages']} pages")
            print(f"  - Found {extracted_data['metadata']['total_images']} images")
            
            # Link entities to images
            linker = EntityImageLinker()
            entity_image_map = linker.link_entities_to_images(extracted_data)
            
            # Save mapping
            mapping_path = os.path.join(mappings_output, f"{pdf_path.stem}_entity_image_map.json")
            linker.save_mapping(mapping_path)
            
            summary = linker.get_entity_summary()
            print(f"  - Extracted {summary['total_entities']} entities")
            print(f"  - {summary['entities_with_images']} entities linked to images")
            print(f"  - Total entity-image links: {summary['total_entity_image_links']}")
            print(f"  - Mapping saved to: {mapping_path}")
            
            results.append({
                'pdf': str(pdf_path),
                'extracted_data': extracted_data,
                'entity_image_map': entity_image_map,
                'mapping_path': mapping_path,
                'summary': summary
            })
            
            print()
            
        except Exception as e:
            print(f"  ERROR: Failed to process {pdf_path.name}: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    # Print overall summary
    print("=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Successfully processed {len(results)} PDF file(s)")
    
    total_images = sum(r['extracted_data']['metadata']['total_images'] for r in results)
    total_entities = sum(r['summary']['total_entities'] for r in results)
    total_links = sum(r['summary']['total_entity_image_links'] for r in results)
    
    print(f"Total images extracted: {total_images}")
    print(f"Total entities found: {total_entities}")
    print(f"Total entity-image links created: {total_links}")
    print()
    print(f"Images directory: {images_output}")
    print(f"Mappings directory: {mappings_output}")
    
    return results


if __name__ == "__main__":
    # Determine default input folder relative to the project root
    default_folder = _resolve_path('data')

    # CLI: an explicit path can be provided; if relative, resolve against project root
    if len(sys.argv) > 1:
        input_folder = _resolve_path(sys.argv[1])
    else:
        input_folder = default_folder

    print(f"Processing PDFs in: {input_folder}")
    print()

    results = process_pdfs_in_folder(input_folder)

    if results:
        print("\nDone! You can now run the NodeRAG build process.")
        print("The entity-image mappings will be loaded automatically when you enable the search engine.")
