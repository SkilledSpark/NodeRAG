import os
import json

from ...config import NodeConfig
from ...storage.storage import storage
from ..component.document import document
from ...logging import info_timer


class document_pipline():

    def __init__(self, config:NodeConfig):
        

        self.config = config
        self.documents_path = self.load_document_path()
        self.indices = self.config.indices
        self._documents = None
        self._hash_ids = None
        self._human_readable_id = None
        
        # Initialize PDF extractor if needed
        self.pdf_extractor = None
        try:
            from ...utils.pdf_extractor import PDFExtractor
            images_dir = os.path.join(self.config.main_folder, 'images')
            self.pdf_extractor = PDFExtractor(images_dir)
        except Exception as e:
            self.config.console.print(f'[yellow]Warning: PDF extractor not available: {e}[/yellow]')
        
    def integrity_check(self):
        if not os.path.exists(self.config.cache):
            os.makedirs(self.config.cache)
        elif self.cache_completion_check():
            pass
        else:
            self.delete_cache()
        
    def load_document_path(self):
        with open(self.config.document_hash_path,'r') as f:
            return json.load(f)['document_path']
            
    @property
    def documents(self):
        if self._documents is None:
            self._documents = []
            for path in self.documents_path:
                # Check if it's a PDF file
                if path.lower().endswith('.pdf'):
                    if self.pdf_extractor:
                        try:
                            extracted_data = self.pdf_extractor.extract(path)
                            raw_context = extracted_data['text']
                            # Store image references in document metadata
                            image_refs = extracted_data.get('images', [])
                        except Exception as e:
                            self.config.console.print(f'[red]Error extracting PDF {path}: {e}[/red]')
                            raw_context = f"Error extracting PDF: {e}"
                            image_refs = []
                        doc = document(raw_context, path, self.config.semantic_text_splitter)
                        doc.image_references = image_refs  # Store image references
                        self._documents.append(doc)
                    else:
                        # Fallback: just read as text (won't work well but prevents crash)
                        self.config.console.print(f'[yellow]PDF extractor not available, skipping PDF: {path}[/yellow]')
                else:
                    # Regular text file
                    with open(path, 'r', encoding='utf-8') as f:
                        raw_context = f.read()
                    self._documents.append(document(raw_context,path,self.config.semantic_text_splitter))
        return self._documents
    
    @property
    def hash_ids(self):
        if not self._hash_ids:
            self._hash_ids = [doc.hash_id for doc in self.documents]
        return self._hash_ids
    
    @property
    def human_readable_ids(self):
        if not self._human_readable_id:
            self._human_readable_id = [doc.human_readable_id for doc in self.documents]
        return self._human_readable_id
    
    
    def store_documents_data(self):
        doc_list = []
        for doc in self.documents:
            doc_list.append({'doc_id':doc.human_readable_id,
                             'doc_hash_id':doc.hash_id,
                             'text_id':doc.text_human_readable_id,
                             'text_hash_id':doc.text_hash_id,
                             'path':doc.path})
        storage(doc_list).save_parquet(self.config.documents_path,append= os.path.exists(self.config.documents_path))
        self.config.console.print('[green]Documents stored[/green]')
        
    def store_text_data(self):
        text_list = []
        
        self.config.tracker.set(len(self.documents),desc="Processing text")
        for doc in self.documents:
            doc.split()
            # Get image references for this document
            image_refs = getattr(doc, 'image_references', [])
            image_refs_dict = {img.get('page', 0): img.get('path', '') for img in image_refs}
            
            for idx, text in enumerate(doc.text_units):
                # Try to associate images with text segments
                associated_images = []
                # Simple heuristic: associate images from same page
                # In practice, you'd want more sophisticated matching
                for img_ref in image_refs:
                    if img_ref.get('page', 0) > 0:  # If page info available
                        associated_images.append(img_ref.get('path', ''))
                
                text_entry = {'text_id':text.human_readable_id,
                            'hash_id':text.hash_id,
                            'type':'text',
                            'context':text.raw_context,
                            'doc_id':doc.human_readable_id,
                            'doc_hash_id':doc.hash_id,
                            'embedding':None,
                            'image_references': json.dumps(associated_images) if associated_images else None}
                text_list.append(text_entry)
            self.config.tracker.update()
        self.config.tracker.close()
        storage(text_list).save_parquet(self.config.text_path,append= os.path.exists(self.config.text_path))
        self.config.console.print('[green]Texts stored[/green]')
        
    def store_readable_index(self) -> None:
        
        self.indices.store_all_indices(self.config.indices_path)
        
    def cache_completion_check(self) -> bool:
        files_name = ['documents.parquet','text.parquet','indices.json']
        files = os.listdir(self.config.cache)
        return all([file in files for file in files_name])
    
    def delete_cache(self) -> None:
        for file in os.listdir(self.config.cache):
            os.remove(os.path.join(self.config.cache,file))    
        self.config.console.print('[red]There exist incomplete cache,deleted[/red]')
        
    def increment_doc(self) -> None:
        if os.path.exists(self.config.documents_path):
            exist_doc_id = storage.load_parquet(self.config.documents_path)['doc_hash_id'].tolist()
            increment_doc_id = list(set(self.hash_ids) - set(exist_doc_id))
            self._documents = [doc for doc in self.documents if doc.hash_id in increment_doc_id]
        else:
            self._documents = self.documents
        self.documents_path = [doc.path for doc in self.documents]
        self._hash_ids = None
        
    
    @info_timer(message='Document Pipeline')
    async def main(self):
        self.integrity_check()
        self.increment_doc()
        self.store_text_data()
        self.store_documents_data()
        self.store_readable_index()
            
        
        
        
        
        
        
        
