"""
PDF Extractor Module for Multimodal Document Processing
Supports text and image extraction from PDF files using MinerU or PyMuPDF
"""
import os
import json
from typing import List, Dict, Tuple, Optional
from pathlib import Path

MINERU_AVAILABLE = False
PYPDF_AVAILABLE = False

# Try to import MinerU
try:
    # MinerU might be imported differently, adjust as needed
    import mineru
    MINERU_AVAILABLE = True
except ImportError:
    pass

# Try to import PyMuPDF (fitz)
if not MINERU_AVAILABLE:
    try:
        import fitz  # PyMuPDF
        PYPDF_AVAILABLE = True
    except ImportError:
        pass


class PDFExtractor:
    """Extract text and images from PDF files"""
    
    def __init__(self, output_dir: str, use_mineru: bool = True, base_path: str = None):
        """
        Initialize PDF extractor
        
        Args:
            output_dir: Directory to save extracted images
            use_mineru: Whether to use MinerU (if available) or fallback to pypdf2
            base_path: Optional base path for computing relative paths (default: output_dir parent)
        """
        self.output_dir = Path(output_dir)
        # If the provided output_dir already points to an 'images' folder,
        # don't create a nested 'images/images' directory.
        if self.output_dir.name.lower() == "images":
            self.images_dir = self.output_dir
        else:
            self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.use_mineru = use_mineru and MINERU_AVAILABLE
        self.base_path = Path(base_path) if base_path else self.output_dir.parent
        
        if not self.use_mineru and not PYPDF_AVAILABLE:
            raise ImportError("Neither MinerU nor PyPDF2/PyMuPDF/pdf2image are available. Please install one of them.")
    
    def extract(self, pdf_path: str) -> Dict:
        """
        Extract text and images from PDF
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with 'text', 'images', and 'metadata' keys
        """
        if self.use_mineru:
            return self._extract_with_mineru(pdf_path)
        else:
            return self._extract_with_pypdf(pdf_path)
    
    def _extract_with_mineru(self, pdf_path: str) -> Dict:
        """Extract using MinerU"""
        # Note: MinerU API may vary, adjust as needed
        try:
            mineru_instance = mineru.MinerU()
            result = mineru_instance.extract(pdf_path)
        except Exception:
            # Fallback to PyMuPDF if MinerU fails
            return self._extract_with_pypdf(pdf_path)
        
        # Extract text
        text_content = []
        images_info = []
        
        # Adjust based on actual MinerU API structure
        try:
            for page_idx, page in enumerate(result.pages):
                # Extract text from page
                page_text = ""
                if hasattr(page, 'elements'):
                    for element in page.elements:
                        if hasattr(element, 'text'):
                            page_text += element.text + "\n"
                elif hasattr(page, 'text'):
                    page_text = page.text
                
                text_content.append({
                    'page': page_idx + 1,
                    'text': page_text
                })
                
                # Extract images
                if hasattr(page, 'images'):
                    for img_idx, img in enumerate(page.images):
                        img_path = self.images_dir / f"{Path(pdf_path).stem}_page{page_idx+1}_img{img_idx+1}.png"
                        if hasattr(img, 'save'):
                            img.save(img_path)
                        elif isinstance(img, bytes):
                            with open(img_path, 'wb') as f:
                                f.write(img)
                        images_info.append({
                            'page': page_idx + 1,
                            'path': str(img_path),
                            'index': img_idx + 1
                        })
        except Exception:
            # If MinerU structure is different, fallback
            return self._extract_with_pypdf(pdf_path)
        
        # Combine all text
        full_text = "\n\n".join([f"Page {p['page']}:\n{p['text']}" for p in text_content])
        
        return {
            'text': full_text,
            'pages': text_content,
            'images': images_info,
            'metadata': {
                'total_pages': len(text_content),
                'total_images': len(images_info),
                'extraction_method': 'MinerU'
            }
        }
    
    def _extract_with_pypdf(self, pdf_path: str) -> Dict:
        """Extract using PyPDF2/PyMuPDF as fallback"""
        import fitz  # PyMuPDF
        
        doc = fitz.open(pdf_path)
        text_content = []
        images_info = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Extract text and block geometry
            page_text = page.get_text()
            page_raw = None
            text_blocks = []
            image_blocks = []
            try:
                page_raw = page.get_text("rawdict")
                for block in page_raw.get('blocks', []):
                    btype = block.get('type')
                    bbox = block.get('bbox')
                    if btype == 0:  # text block
                        # Concatenate spans text to represent the block
                        block_text_parts = []
                        for line in block.get('lines', []):
                            for span in line.get('spans', []):
                                t = span.get('text', '')
                                if t:
                                    block_text_parts.append(t)
                        block_text = ''.join(block_text_parts)
                        text_blocks.append({'bbox': bbox, 'text': block_text})
                    elif btype == 1:  # image block
                        image_blocks.append({'bbox': bbox})
            except Exception:
                # rawdict may not be available in some cases; proceed without geometry
                pass
            
            text_content.append({
                'page': page_num + 1,
                'text': page_text,
                'text_blocks': text_blocks,
                'image_blocks': image_blocks
            })
            
            # Extract images
            image_list = page.get_images()
            # Attempt to align image blocks (with bbox) to extracted images by index order
            img_block_bboxes = []
            if text_content[-1].get('image_blocks'):
                img_block_bboxes = [blk['bbox'] for blk in text_content[-1]['image_blocks']]
            for img_idx, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    img_path = self.images_dir / f"{Path(pdf_path).stem}_page{page_num+1}_img{img_idx+1}.{image_ext}"
                    with open(img_path, "wb") as img_file:
                        img_file.write(image_bytes)
                    
                    # Store relative path if base_path is set
                    stored_path = str(img_path)
                    try:
                        if self.base_path:
                            stored_path = os.path.relpath(img_path, self.base_path)
                    except ValueError:
                        # relpath fails if paths are on different drives on Windows
                        pass
                    
                    images_info.append({
                        'page': page_num + 1,
                        'path': stored_path,
                        'index': img_idx + 1,
                        'bbox': img_block_bboxes[img_idx] if img_idx < len(img_block_bboxes) else None
                    })
                except Exception as e:
                    print(f"Warning: Could not extract image {img_idx} from page {page_num + 1}: {e}")
        
        doc.close()
        
        # Combine all text
        full_text = "\n\n".join([f"Page {p['page']}:\n{p['text']}" for p in text_content])
        
        return {
            'text': full_text,
            'pages': text_content,
            'images': images_info,
            'metadata': {
                'total_pages': len(text_content),
                'total_images': len(images_info),
                'extraction_method': 'PyMuPDF'
            }
        }
    
    def get_image_references(self, extracted_data: Dict, text_segments: List[str]) -> List[Dict]:
        """
        Create references from text segments to nearby images
        
        Args:
            extracted_data: Output from extract() method
            text_segments: List of text segments (e.g., from text splitting)
            
        Returns:
            List of dictionaries with text_segment_index and image_path mappings
        """
        references = []
        images_by_page = {}
        
        # Group images by page
        for img in extracted_data['images']:
            page = img['page']
            if page not in images_by_page:
                images_by_page[page] = []
            images_by_page[page].append(img)
        
        # For each text segment, find which page it likely came from
        # This is a simple heuristic - in practice, you'd want more sophisticated matching
        for seg_idx, segment in enumerate(text_segments):
            # Try to find page numbers mentioned in segment
            page_refs = []
            for page_info in extracted_data['pages']:
                if str(page_info['page']) in segment or f"Page {page_info['page']}" in segment:
                    page_refs.append(page_info['page'])
            
            # If no explicit page reference, use page 1 as default (can be improved)
            if not page_refs:
                page_refs = [1]
            
            # Link to images on those pages
            for page_num in page_refs:
                if page_num in images_by_page:
                    for img in images_by_page[page_num]:
                        references.append({
                            'text_segment_index': seg_idx,
                            'image_path': img['path'],
                            'page': page_num,
                            'image_index': img['index']
                        })
        
        return references

