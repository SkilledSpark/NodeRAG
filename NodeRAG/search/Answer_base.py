from ..config import NodeConfig

class Retrieval():
    
    def __init__(self,config:NodeConfig,id_to_text:dict,accurate_id_to_text:dict,id_to_type:dict):
        
        self.config = config
        self.HNSW_results_with_distance = None
        self._HNSW_results = None
        self.id_to_text = id_to_text
        self.accurate_id_to_text = accurate_id_to_text
        self.accurate_results = None
        self.search_list = []
        self.unique_search_list = set()
        self.id_to_type = id_to_type
        self.relationship_list = None
        self._retrieved_list = None
        self._structured_prompt = None
        self._unstructured_prompt = None
        self.associated_images = []
        self._entity_images = None
        
        
        
    @property
    def HNSW_results(self):
        if self._HNSW_results is None:
            self._HNSW_results = [id for distance,id in self.HNSW_results_with_distance]
            self.search_list.extend(self._HNSW_results)
            self.unique_search_list.update(self._HNSW_results)
        return self._HNSW_results
    
    @property
    def model_name(self):
        return self.config.API_client.llm.model_name
    
    @property
    def HNSW_results_str(self):
        return [self.id_to_text[id] for id in self.HNSW_results]
    
    @property
    def accurate_results_str(self):
        return [self.accurate_id_to_text[id] for id in self.accurate_results]
    
    @property
    def retrieved_list(self):
        if self._retrieved_list is None:
            self._retrieved_list = [(self.id_to_text[id],self.id_to_type[id]) for id in self.search_list]+ [(self.id_to_text[id],'relationship') for id in self.relationship_list]
        return self._retrieved_list
    
    @property
    def structured_prompt(self):
        if self._structured_prompt is None:
            self._structured_prompt = self.types_info()
        return self._structured_prompt
    
    @property
    def unstructured_prompt(self)->str:
        if self._unstructured_prompt is None:
            self._unstructured_prompt = '\n'.join([content for content,_ in self.retrieved_list])
        return self._unstructured_prompt
    
    @property
    def retrieval_info(self)->str:
        return self.structured_prompt
    
    def types_info(self)->str:
        types = set([type for _,type in self.retrieved_list])
        prompt = ''
        for type in types:
            prompt += f'------------{type}-------------\n'
            n=1
            for content,typed in self.retrieved_list:
                if typed == type:
                    prompt += f'{n}. {content}\n'
                    n+=1
            prompt += '\n\n'
        return prompt
    
    def add_image_association(self, image_path: str, description: str = None, entities: list = None) -> None:
        """Add an image association to the retrieval results"""
        image_info = {
            'path': image_path,
            'description': description,
            'entities': entities or []
        }
        self.associated_images.append(image_info)
    
    @property
    def entity_images(self) -> dict:
        """Get images organized by entity"""
        if self._entity_images is None:
            self._entity_images = {}
            for img in self.associated_images:
                for entity in img.get('entities', []):
                    if entity not in self._entity_images:
                        self._entity_images[entity] = []
                    self._entity_images[entity].append(img)
        return self._entity_images
    
    def get_images_for_query(self, query_entities: list = None) -> list:
        """Get relevant images based on query entities"""
        if not query_entities:
            return self.associated_images
        
        relevant_images = []
        for img in self.associated_images:
            img_entities = [e.lower() for e in img.get('entities', [])]
            if any(entity.lower() in img_entities for entity in query_entities):
                relevant_images.append(img)
        return relevant_images
    
    def __str__(self):
        return self.retrieval_info
    
    
    
class Answer():
    
    def __init__(self,query:str,retrieval:Retrieval):
        self.query = query
        self.retrieval = retrieval
        self.response = None
        self._relevant_images = None
        
    @property
    def retrieval_info(self):
        return self.retrieval.retrieval_info
    
    @property
    def structured_prompt(self):
        return self.retrieval.structured_prompt
    
    @property
    def unstructured_prompt(self):
        return self.retrieval.unstructured_prompt
    
    @property
    def retrieval_tokens(self):
        return self.retrieval.config.token_counter(self.retrieval_info)
    
    @property
    def response_tokens(self):
        return self.retrieval.config.token_counter(self.response)
    
    @property
    def relevant_images(self) -> list:
        """Get images relevant to the query"""
        if self._relevant_images is None:
            # Extract entities from query for image matching
            query_entities = self._extract_entities_from_query()
            self._relevant_images = self.retrieval.get_images_for_query(query_entities)
        return self._relevant_images
    
    def _extract_entities_from_query(self) -> list:
        """Extract potential entities from the query for image matching"""
        # Simple entity extraction - can be enhanced with NER
        import re
        # Look for capitalized words that might be entities
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', self.query)
        return entities
    
    def get_images_info(self) -> str:
        """Get formatted information about relevant images"""
        if not self.relevant_images:
            return ""
        
        images_info = "\n\n------------RELATED IMAGES-------------\n"
        for i, img in enumerate(self.relevant_images, 1):
            images_info += f"{i}. Image: {img['path']}\n"
            if img.get('description'):
                images_info += f"   Description: {img['description']}\n"
            if img.get('entities'):
                images_info += f"   Related to: {', '.join(img['entities'])}\n"
            images_info += "\n"
        return images_info
    
    def get_full_response_with_images(self) -> dict:
        """Get response with both text and image information"""
        return {
            'text_response': self.response,
            'images': self.relevant_images,
            'images_info': self.get_images_info()
        }
    
    def __str__(self):
        return self.response
    

