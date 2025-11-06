import os
from typing import Dict,List,Tuple
import numpy as np
import re


from ..storage import Mapper
from ..utils import HNSW
from ..storage import storage
from ..utils.graph_operator import GraphConcat
from ..config import NodeConfig
from ..utils.PPR import sparse_PPR
from .Answer_base import Answer,Retrieval




class NodeSearch():

    def __init__(self,config:NodeConfig):
        

        self.config = config
        self.hnsw = self.load_hnsw()
        self.mapper = self.load_mapper()
        self.G = self.load_graph()
        self.id_to_type = {id:self.G.nodes[id].get('type') for id in self.G.nodes}
        self.id_to_text,self.accurate_id_to_text = self.mapper.generate_id_to_text(['entity','high_level_element_title'])
        self.sparse_PPR = sparse_PPR(self.G)
        self._semantic_units = None
        self.image_registry = {}  # Registry to store image associations
        # Load image associations from mappings and directories if enabled
        if getattr(self.config, 'enable_images', True):
            try:
                self._load_entity_image_mappings()
                # Also scan image directories to maximize recall (optional)
                base_dir = self._get_base_data_dir()
                extracted_dir = os.path.join(base_dir, 'extracted_images')
                self.load_images_from_directory(extracted_dir)
            except Exception as e:
                # Non-fatal: image loading shouldn't break search
                print(f"[ImageLoader] Warning: failed to load images: {e}")
            
        
    def load_mapper(self) -> Mapper:
        
        mapping_list = [self.config.semantic_units_path,
                        self.config.entities_path,
                        self.config.relationship_path,
                        self.config.attributes_path,
                        self.config.high_level_elements_path,
                        self.config.text_path,
                        self.config.high_level_elements_titles_path]
        
        for path in mapping_list:
            if not os.path.exists(path):
                raise Exception(f'{path} not found, Please check cache integrity. You may need to rebuild the database due to the loss of cache files.')
        
        mapper = Mapper(mapping_list)
        
        return mapper
    
    def load_hnsw(self) -> HNSW:
        if os.path.exists(self.config.HNSW_path):
            hnsw = HNSW(self.config)
            hnsw.load_HNSW()
            return hnsw
        else:
            raise Exception('No HNSW data found.')
        
    def load_graph(self):
        
        if os.path.exists(self.config.base_graph_path):
            G = storage.load(self.config.base_graph_path)
        else:
            raise Exception('No base graph found.')
        
        if os.path.exists(self.config.hnsw_graph_path):
            HNSW_graph = storage.load(self.config.hnsw_graph_path)
        else:
            raise Exception('No HNSW graph found.')
        
        if self.config.unbalance_adjust:
                G = GraphConcat(G).concat(HNSW_graph)
                return GraphConcat.unbalance_adjust(G)
            
        return GraphConcat(G).concat(HNSW_graph)
        
    
    def search(self,query:str):
        
        retrieval = Retrieval(self.config,self.id_to_text,self.accurate_id_to_text,self.id_to_type)
        

        # HNSW search for enter points by cosine similarity
        query_embedding = np.array(self.config.embedding_client.request(query),dtype=np.float32)
        HNSW_results = self.hnsw.search(query_embedding,HNSW_results=self.config.HNSW_results)
        retrieval.HNSW_results_with_distance = HNSW_results
        
        
        
        # Decompose query into entities and accurate search for short words level items.
        decomposed_entities = self.decompose_query(query)
        
        accurate_results = self.accurate_search(decomposed_entities)
        retrieval.accurate_results = accurate_results
        
        # Personlization for graph search
        personlization = {ids:self.config.similarity_weight for ids in retrieval.HNSW_results}
        personlization.update({id:self.config.accuracy_weight for id in retrieval.accurate_results})
        
        weighted_nodes = self.graph_search(personlization)
        
        retrieval = self.post_process_top_k(weighted_nodes,retrieval)
        
        # Add image associations to retrieval
        self.add_images_to_retrieval(retrieval, decomposed_entities)

        return retrieval

    def decompose_query(self,query:str):
        
        query = self.config.prompt_manager.decompose_query.format(query=query)
        response = self.config.API_client.request({'query':query,'response_format':self.config.prompt_manager.decomposed_text_json})
        return response['elements']
    
    
    def accurate_search(self, entities: List[str]) -> List[str]:
        accurate_results = []
        
        for entity in entities:
            # Split entity into words and create a pattern to match the whole phrase
            words = entity.lower().split()
            pattern = re.compile(r'\b' + r'\s+'.join(map(re.escape, words)) + r'\b')
            result = [id for id, text in self.accurate_id_to_text.items() if pattern.search(text.lower())]
            if result:
                accurate_results.extend(result)
        
        return accurate_results
    
    
    def answer(self,query:str,id_type:bool=True):
        
        
        retrieval = self.search(query)
        
        ans = Answer(query,retrieval)
        
        if id_type:
            retrieved_info = ans.structured_prompt
        else:
            retrieved_info = ans.unstructured_prompt
        
        query = self.config.prompt_manager.answer.format(info=retrieved_info,query=query)
        ans.response = self.config.API_client.request({'query':query})
        
        return ans
    
    
    
    async def answer_async(self,query:str,id_type:bool=True):
        
        
        retrieval = self.search(query)
        
        ans = Answer(query,retrieval)
        
        if id_type:
            retrieved_info = ans.structured_prompt
        else    :
            retrieved_info = ans.unstructured_prompt

        query = self.config.prompt_manager.answer.format(info=retrieved_info,query=query)
        
        ans.response = await self.config.API_client({'query':query})
        
        return ans
        
    
    def stream_answer(self, query: str, retrieved_info: str, system_prompt: str | None = None):
        """
        Stream answer with proper system/user prompt separation for OpenAI models.
        
        Args:
            query: The user's question
            retrieved_info: The retrieved context information
            system_prompt: Optional custom system prompt (overrides default template system prompt)
        """
        # Get the answer prompt template
        answer_template = self.config.prompt_manager.answer
        
        # Format the full prompt with retrieved info and query
        formatted_prompt = answer_template.format(info=retrieved_info, query=query)
        
        # For OpenAI models, separate system and user prompts
        if system_prompt:
            # When custom system prompt is provided, use it as the ONLY system prompt
            # and format the user content with retrieved context and query
            parts = formatted_prompt.split("---Retrived Context---", 1)
            if len(parts) == 2:
                # Extract just the context and query parts (without the template's system instructions)
                user_content = "---Retrived Context---" + parts[1]
                response = self.config.API_client.stream_chat({
                    'system_prompt': system_prompt,
                    'query': user_content
                })
            else:
                # Fallback: use custom system + full formatted as query
                response = self.config.API_client.stream_chat({
                    'system_prompt': system_prompt,
                    'query': formatted_prompt
                })
        else:
            # No custom system prompt - separate the template's role section
            parts = formatted_prompt.split("---Retrived Context---", 1)
            if len(parts) == 2:
                # Use Role/Goal/Format sections as system prompt
                system_part = parts[0].strip()
                user_part = "---Retrived Context---" + parts[1]
                response = self.config.API_client.stream_chat({
                    'system_prompt': system_part,
                    'query': user_part
                })
            else:
                # Fallback to original behavior
                response = self.config.API_client.stream_chat({'query': formatted_prompt})
        
        yield from response

    # -------- Images: load + normalize paths --------

    def _get_base_data_dir(self) -> str:
        """Return the base data directory (parent of 'input' if main_folder points to it)."""
        mf = self.config.main_folder  # absolute
        if os.path.basename(mf.rstrip('/\\')).lower() == 'input':
            return os.path.dirname(mf)
        return mf

    def _normalize_image_path(self, relative_or_abs_path: str) -> str:
        """Resolve image path to an absolute existing path when possible.

        Accepts values like '<pdf_stem>/images/file.jpeg' and returns
        '<base>/extracted_images/<pdf_stem>/images/file.jpeg' if that exists.
        """
        p = relative_or_abs_path.replace('\\', os.sep).replace('/', os.sep)
        # Quick fix for accidental 'images/images'
        p = p.replace(os.sep + 'images' + os.sep + 'images' + os.sep, os.sep + 'images' + os.sep)

        if os.path.isabs(p) and os.path.exists(p):
            return p

        base_dir = self._get_base_data_dir()
        candidates = [
            os.path.join(base_dir, 'extracted_images', p),
            os.path.join(base_dir, p),  # in case mapping already included 'extracted_images/...'
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        # Fallback: return first candidate (absolute) even if missing, UI may handle missing gracefully
        return candidates[0]

    def _load_entity_image_mappings(self) -> None:
        """Load entity->images mapping JSONs and register images in the registry."""
        base_dir = self._get_base_data_dir()
        mappings_dir = os.path.join(base_dir, 'entity_image_mappings')
        if not os.path.exists(mappings_dir):
            return

        try:
            from pathlib import Path
            for mp in Path(mappings_dir).glob('*.json'):
                try:
                    import json
                    with open(mp, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    for key, val in data.items():
                        name = val.get('name') or key
                        label = val.get('label')
                        images = val.get('images') or []
                        norm_paths = [self._normalize_image_path(ip) for ip in images]
                        # Register each image with the entity
                        for ip in norm_paths:
                            self.register_image(ip, description=f"{label or 'entity'} image", entities=[name])
                except Exception as e:
                    print(f"[ImageLoader] Skipped mapping {mp.name}: {e}")
        except Exception as e:
            print(f"[ImageLoader] Failed scanning mappings: {e}")


    def graph_search(self,personlization:Dict[str,float])->List[Tuple[str,str]]|List[str]:
        
        page_rank_scores = self.sparse_PPR.PPR(personlization,alpha=self.config.ppr_alpha,max_iter=self.config.ppr_max_iter)
        
        
        return [id for id,score in page_rank_scores]
        
    
    def post_process_top_k(self,weighted_nodes:List[str],retrieval:Retrieval)->Retrieval:
        
        
        entity_list = []
        high_level_element_title_list = []
        relationship_list = []
    
        addition_node = 0
        
        for node in weighted_nodes:
            if node not in retrieval.search_list:
                type = self.G.nodes[node].get('type')
                match type:
                    case 'entity':
                        if node not in entity_list and len(entity_list) < self.config.Enode:
                            entity_list.append(node)
                    case 'relationship':
                        if node not in relationship_list and len(relationship_list) < self.config.Rnode:
                            relationship_list.append(node)
                    case 'high_level_element_title':
                        if node not in high_level_element_title_list and len(high_level_element_title_list) < self.config.Hnode:
                            high_level_element_title_list.append(node)
        
                    case _:
                        if addition_node < self.config.cross_node:
                            if node not in retrieval.unique_search_list:
                                retrieval.search_list.append(node)
                                retrieval.unique_search_list.add(node)
                                addition_node += 1
                
                if (addition_node >= self.config.cross_node 
                    and len(entity_list) >= self.config.Enode  
                    and len(relationship_list) >= self.config.Rnode 
                    and len(high_level_element_title_list) >= self.config.Hnode):
                    break
        
        for entity in entity_list:
            attributes = self.G.nodes[entity].get('attributes')
            if attributes:
                for attribute in attributes:
                    if attribute not in retrieval.unique_search_list:
                        retrieval.search_list.append(attribute)
                        retrieval.unique_search_list.add(attribute)

    

        for high_level_element_title in high_level_element_title_list:
            related_node = self.G.nodes[high_level_element_title].get('related_node')
            if related_node not in retrieval.unique_search_list:
                retrieval.search_list.append(related_node)
                retrieval.unique_search_list.add(related_node)
            
            
        
        retrieval.relationship_list = list(set(relationship_list))
        
        return retrieval
    
    def register_image(self, image_path: str, description: str = None, entities: list = None, document_id: str = None) -> None:
        """Register an image with associated entities and metadata"""
        image_info = {
            'path': image_path,
            'description': description,
            'entities': entities or [],
            'document_id': document_id
        }
        
        # Store by image path
        self.image_registry[image_path] = image_info
        
        # Also index by entities for quick lookup
        for entity in entities or []:
            entity_key = f"entity:{entity.lower()}"
            if entity_key not in self.image_registry:
                self.image_registry[entity_key] = []
            self.image_registry[entity_key].append(image_info)
    
    def add_images_to_retrieval(self, retrieval: Retrieval, query_entities: list) -> None:
        """Add relevant images to the retrieval results"""
        # Find images associated with query entities
        for entity in query_entities:
            entity_key = f"entity:{entity.lower()}"
            if entity_key in self.image_registry:
                for image_info in self.image_registry[entity_key]:
                    retrieval.add_image_association(
                        image_info['path'],
                        image_info['description'],
                        image_info['entities']
                    )
        
        # Also check retrieved entities for image associations
        for node_id in retrieval.search_list:
            node_text = self.id_to_text.get(node_id, "")
            node_type = self.id_to_type.get(node_id, "")
            
            if node_type == 'entity':
                # Extract entity name from text for image lookup
                entity_name = node_text.split(':')[0] if ':' in node_text else node_text
                entity_key = f"entity:{entity_name.lower()}"
                if entity_key in self.image_registry:
                    for image_info in self.image_registry[entity_key]:
                        retrieval.add_image_association(
                            image_info['path'],
                            image_info['description'],
                            image_info['entities']
                        )
    
    def load_images_from_directory(self, directory_path: str, entity_mapping: dict = None) -> None:
        """Load images from a directory and associate them with entities"""
        import os
        from pathlib import Path
        
        if not os.path.exists(directory_path):
            return
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg', '.tiff', '.ico'}
        
        for file_path in Path(directory_path).rglob('*'):
            if file_path.suffix.lower() in image_extensions:
                # Extract potential entity names from filename
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