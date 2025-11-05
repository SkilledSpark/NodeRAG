"""
Entity-Image Linking Module
Links extracted entities from text with images in PDF documents
"""
import os
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import re

try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
        NLP_AVAILABLE = True
    except:
        NLP_AVAILABLE = False
        nlp = None
except ImportError:
    NLP_AVAILABLE = False
    nlp = None


class EntityImageLinker:
    """Links entities extracted from text with images from PDFs"""
    
    def __init__(self, image_handler=None):
        """
        Initialize the entity-image linker
        
        Args:
            image_handler: Optional ImageHandler instance for image processing
        """
        self.image_handler = image_handler
        self.entity_image_map = {}
        self.nlp = nlp if NLP_AVAILABLE else None
    
    def extract_entities_from_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities from text using spaCy
        
        Args:
            text: Input text
            
        Returns:
            List of entity dictionaries with 'text', 'label', and 'start'/'end' positions
        """
        if not self.nlp:
            # Fallback: simple capitalized word extraction
            entities = []
            for match in re.finditer(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text):
                entities.append({
                    'text': match.group(),
                    'label': 'UNKNOWN',
                    'start': match.start(),
                    'end': match.end()
                })
            return entities
        
        doc = self.nlp(text)
        entities = []
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
        return entities
    
    def link_entities_to_images(
        self,
        extracted_pdf_data: Dict,
        proximity_threshold: Optional[float] = None,
        max_images_per_entity: int = 1
    ) -> Dict[str, List[str]]:
        """
        Link entities found in PDF text to nearest images on the same page using layout geometry when available
        
        Args:
            extracted_pdf_data: Data from PDFExtractor.extract()
            proximity_threshold: Optional max center-to-center distance (page units) to accept a link. If None, always pick nearest.
            max_images_per_entity: Maximum number of images to associate per entity (default 1)
            
        Returns:
            Dictionary mapping entity names to list of image paths
        """
        entity_image_map = {}
        
        def _center(bbox):
            if not bbox or len(bbox) != 4:
                return None
            x0, y0, x1, y1 = bbox
            return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
        
        def _dist(c1, c2):
            return ((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2) ** 0.5
        
        # Process each page
        for page_info in extracted_pdf_data.get('pages', []):
            page_num = page_info['page']
            page_text = page_info['text']
            text_blocks = page_info.get('text_blocks') or []
            
            # Extract entities from this page's text
            entities = self.extract_entities_from_text(page_text)
            
            # Get images from this page
            page_images_full = [
                img for img in extracted_pdf_data.get('images', [])
                if img['page'] == page_num
            ]
            
            # Link entities to nearest images on the same page
            for entity in entities:
                entity_name = entity['text']
                entity_key = entity_name.lower()
                
                if entity_key not in entity_image_map:
                    entity_image_map[entity_key] = {
                        'name': entity_name,
                        'label': entity['label'],
                        'images': [],
                        'pages': set()
                    }
                
                # Find the text block that contains this entity (best-effort)
                matched_block_center = None
                for blk in text_blocks:
                    bt = blk.get('text') or ''
                    if bt and entity_name.lower() in bt.lower():
                        matched_block_center = _center(blk.get('bbox'))
                        if matched_block_center:
                            break
                
                chosen_images = []
                # Prefer geometry-based nearest selection when image bbox exists
                images_with_centers = []
                for img in page_images_full:
                    c = _center(img.get('bbox')) if img.get('bbox') else None
                    if c is not None:
                        images_with_centers.append((img, c))
                
                if matched_block_center and images_with_centers:
                    # Rank by distance to the matched text block center
                    ranked = sorted(
                        images_with_centers,
                        key=lambda ic: _dist(matched_block_center, ic[1])
                    )
                    for (img, c) in ranked:
                        d = _dist(matched_block_center, c)
                        if proximity_threshold is None or d <= proximity_threshold:
                            chosen_images.append(img['path'])
                        if len(chosen_images) >= max_images_per_entity:
                            break
                else:
                    # Fallback: no geometry; pick the first few images on the page
                    for img in page_images_full[:max_images_per_entity]:
                        chosen_images.append(img['path'])
                
                # Record chosen images
                for img_path in chosen_images:
                    if img_path not in entity_image_map[entity_key]['images']:
                        entity_image_map[entity_key]['images'].append(img_path)
                
                entity_image_map[entity_key]['pages'].add(page_num)
        
        # Convert sets to lists for JSON serialization
        for entity_key in entity_image_map:
            entity_image_map[entity_key]['pages'] = list(entity_image_map[entity_key]['pages'])
        
        self.entity_image_map = entity_image_map
        return entity_image_map
    
    def find_images_for_entity(self, entity_name: str) -> List[str]:
        """
        Find all images associated with a given entity
        
        Args:
            entity_name: Name of the entity
            
        Returns:
            List of image paths
        """
        entity_key = entity_name.lower()
        if entity_key in self.entity_image_map:
            return self.entity_image_map[entity_key]['images']
        return []
    
    def find_images_for_entities(self, entity_names: List[str]) -> Dict[str, List[str]]:
        """
        Find images for multiple entities
        
        Args:
            entity_names: List of entity names
            
        Returns:
            Dictionary mapping entity names to their image paths
        """
        result = {}
        for entity_name in entity_names:
            images = self.find_images_for_entity(entity_name)
            if images:
                result[entity_name] = images
        return result
    
    def save_mapping(self, output_path: str):
        """
        Save entity-image mapping to JSON file
        
        Args:
            output_path: Path to save the JSON file
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.entity_image_map, f, indent=2, ensure_ascii=False)
    
    def load_mapping(self, mapping_path: str):
        """
        Load entity-image mapping from JSON file
        
        Args:
            mapping_path: Path to the JSON file
        """
        with open(mapping_path, 'r', encoding='utf-8') as f:
            self.entity_image_map = json.load(f)
    
    def get_entity_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics about entity-image mappings
        
        Returns:
            Dictionary with summary statistics
        """
        total_entities = len(self.entity_image_map)
        entities_with_images = sum(
            1 for e in self.entity_image_map.values() if e['images']
        )
        total_links = sum(
            len(e['images']) for e in self.entity_image_map.values()
        )
        
        entity_types = {}
        for entity_data in self.entity_image_map.values():
            label = entity_data.get('label', 'UNKNOWN')
            entity_types[label] = entity_types.get(label, 0) + 1
        
        return {
            'total_entities': total_entities,
            'entities_with_images': entities_with_images,
            'total_entity_image_links': total_links,
            'entity_types': entity_types
        }


def process_pdf_with_entity_linking(
    pdf_path: str,
    output_dir: str,
    image_handler=None
) -> Dict[str, Any]:
    """
    Complete pipeline: Extract PDF, link entities to images
    
    Args:
        pdf_path: Path to PDF file
        output_dir: Directory for output files
        image_handler: Optional ImageHandler instance
        
    Returns:
        Dictionary with extracted data and entity-image mappings
    """
    from .pdf_extractor import PDFExtractor
    
    # Extract PDF content
    extractor = PDFExtractor(output_dir)
    extracted_data = extractor.extract(pdf_path)
    
    # Link entities to images
    linker = EntityImageLinker(image_handler)
    entity_image_map = linker.link_entities_to_images(extracted_data)
    
    # Save mappings
    pdf_name = Path(pdf_path).stem
    mapping_path = os.path.join(output_dir, f"{pdf_name}_entity_image_map.json")
    linker.save_mapping(mapping_path)
    
    # Return comprehensive result
    return {
        'pdf_path': pdf_path,
        'extracted_data': extracted_data,
        'entity_image_map': entity_image_map,
        'mapping_path': mapping_path,
        'summary': linker.get_entity_summary()
    }
