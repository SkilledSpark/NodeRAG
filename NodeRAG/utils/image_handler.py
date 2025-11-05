import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import base64
from PIL import Image
import io


class ImageHandler:
    """Handles image processing and association with entities"""
    
    def __init__(self, config=None):
        self.config = config
        self.image_registry = {}
        self.supported_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg', '.tiff', '.ico'}
        
        if config and hasattr(config, 'image_extensions'):
            self.supported_extensions = set(config.image_extensions)
    
    def register_image(self, image_path: str, description: str = None, entities: List[str] = None, document_id: str = None) -> None:
        """Register an image with associated entities and metadata"""
        if not os.path.exists(image_path):
            return
            
        image_info = {
            'path': image_path,
            'description': description,
            'entities': entities or [],
            'document_id': document_id,
            'size': self._get_image_size(image_path),
            'format': Path(image_path).suffix.lower()
        }
        
        # Store by image path
        self.image_registry[image_path] = image_info
        
        # Also index by entities for quick lookup
        for entity in entities or []:
            entity_key = f"entity:{entity.lower()}"
            if entity_key not in self.image_registry:
                self.image_registry[entity_key] = []
            self.image_registry[entity_key].append(image_info)
    
    def _get_image_size(self, image_path: str) -> Optional[Tuple[int, int]]:
        """Get image dimensions"""
        try:
            with Image.open(image_path) as img:
                return img.size
        except Exception:
            return None
    
    def load_images_from_directory(self, directory_path: str, entity_mapping: Dict[str, List[str]] = None) -> int:
        """Load images from a directory and associate them with entities"""
        if not os.path.exists(directory_path):
            return 0
        
        loaded_count = 0
        
        for file_path in Path(directory_path).rglob('*'):
            if file_path.suffix.lower() in self.supported_extensions:
                filename = file_path.stem
                entities = []
                
                if entity_mapping and filename in entity_mapping:
                    entities = entity_mapping[filename]
                else:
                    # Simple heuristic: split by common separators and capitalize
                    potential_entities = re.split(r'[_\-\s]+', filename)
                    entities = [entity.title() for entity in potential_entities if len(entity) > 2]
                
                self.register_image(
                    str(file_path),
                    f"Image of {', '.join(entities)}" if entities else f"Image: {filename}",
                    entities
                )
                loaded_count += 1
        
        return loaded_count
    
    def find_images_for_entities(self, entities: List[str]) -> List[Dict]:
        """Find images associated with given entities"""
        found_images = []
        seen_paths = set()
        
        for entity in entities:
            entity_key = f"entity:{entity.lower()}"
            if entity_key in self.image_registry:
                for image_info in self.image_registry[entity_key]:
                    if image_info['path'] not in seen_paths:
                        found_images.append(image_info)
                        seen_paths.add(image_info['path'])
        
        return found_images
    
    def get_image_base64(self, image_path: str, max_size: Tuple[int, int] = (800, 600)) -> Optional[str]:
        """Convert image to base64 string for display, with optional resizing"""
        try:
            with Image.open(image_path) as img:
                # Resize if image is too large
                if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                    img.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                # Convert to RGB if necessary
                if img.mode in ('RGBA', 'LA', 'P'):
                    img = img.convert('RGB')
                
                # Save to bytes
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=85)
                img_bytes = buffer.getvalue()
                
                # Encode to base64
                return base64.b64encode(img_bytes).decode('utf-8')
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None
    
    def create_image_html(self, image_info: Dict, include_base64: bool = False) -> str:
        """Create HTML representation of an image"""
        html = '<div class="image-container">\n'
        
        if include_base64:
            base64_data = self.get_image_base64(image_info['path'])
            if base64_data:
                html += f'  <img src="data:image/jpeg;base64,{base64_data}" alt="{image_info.get("description", "Image")}" style="max-width: 400px; max-height: 300px;">\n'
            else:
                html += f'  <p>Image: {image_info["path"]}</p>\n'
        else:
            html += f'  <img src="file://{image_info["path"]}" alt="{image_info.get("description", "Image")}" style="max-width: 400px; max-height: 300px;">\n'
        
        if image_info.get('description'):
            html += f'  <p class="image-description">{image_info["description"]}</p>\n'
        
        if image_info.get('entities'):
            html += f'  <p class="image-entities">Related to: {", ".join(image_info["entities"])}</p>\n'
        
        html += '</div>\n'
        return html
    
    def get_images_summary(self) -> Dict:
        """Get summary statistics about registered images"""
        total_images = len([k for k in self.image_registry.keys() if not k.startswith('entity:')])
        entity_count = len([k for k in self.image_registry.keys() if k.startswith('entity:')])
        
        formats = {}
        for image_info in self.image_registry.values():
            if isinstance(image_info, dict) and 'format' in image_info:
                fmt = image_info['format']
                formats[fmt] = formats.get(fmt, 0) + 1
        
        return {
            'total_images': total_images,
            'entities_with_images': entity_count,
            'formats': formats
        }
