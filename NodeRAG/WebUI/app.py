import streamlit as st
import time
from typing import List,Tuple
import yaml
import os
import json
import sys
import pickle

from NodeRAG.utils import LazyImport

NG = LazyImport('NodeRAG','NodeRag')
NGSearch = LazyImport('NodeRAG','NodeSearch')
NGConfig = LazyImport('NodeRAG','NodeConfig')



# init session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if 'settings' not in st.session_state:
    st.session_state.settings = {}
    st.session_state.settings['relevant_info'] = 'On'
if 'indices' not in st.session_state:
    st.session_state.indices = {}
if 'main_folder' not in st.session_state:
    st.session_state.main_folder = None


args = sys.argv

# Default main folder to user's desired input folder
DEFAULT_MAIN_FOLDER = 'data'
main_folder = DEFAULT_MAIN_FOLDER

# Allow override via --main_folder=...
config_path = None
for arg in args:
    if arg.startswith('--main_folder='):
        main_folder = arg.split('=')[1]

# Use the parent directory of main_folder for config files (if main_folder ends with 'input', use parent)
if os.path.basename(main_folder.rstrip('/\\')).lower() == 'input':
    config_base_folder = os.path.dirname(main_folder)
else:
    config_base_folder = main_folder

st.session_state.original_config_path = os.path.join(config_base_folder, 'Node_config.yaml')
st.session_state.web_ui_config_path = os.path.join(config_base_folder, 'web_ui_config.yaml')



class State_Observer:
    
    def __init__(self,build_status):
        self.build_status = build_status
    
    def update(self,state):
        self.build_status.status(f"🔄 Building Status: {state}")
    
    def reset(self,total_tasks:List[str],desc:str=""):
        if isinstance(total_tasks, list):
            task_str = "\n".join([f"  └─ {task}" for task in total_tasks])
            st.markdown(f"🔄 Building Status: {desc}\nTasks:\n{task_str}")
            time.sleep(2)

    def close(self):
        self.build_status.empty()
        
        
def load_config(path):
    """Load the config from the config file"""
    with open(path, 'r') as file:
        all_config = yaml.safe_load(file)
        st.session_state.config = all_config['config']
        st.session_state.model_config = all_config['model_config']
        st.session_state.embedding_config = all_config['embedding_config']
    
    # Ensure main_folder is set to the default relative path
    DEFAULT_MAIN_FOLDER = 'data'
    if 'main_folder' not in st.session_state.config or st.session_state.config['main_folder'] != DEFAULT_MAIN_FOLDER:
        st.session_state.config['main_folder'] = DEFAULT_MAIN_FOLDER

def all_config():
    """Get all the config from the session state"""
    return {
        'config': st.session_state.config,
        'model_config': st.session_state.model_config,
        'embedding_config': st.session_state.embedding_config
    }
    
def save_config(path):
    """Save the config to the config file"""
    with open(path, 'w') as file:
        yaml.dump(all_config(), file)
    

def display_header():
    """Display the header section with title and description"""
    # Create two columns for title and expander
    col1,= st.columns([0.5])  # Numbers represent the width ratio of the columns

    # Put title in first column
    with col1:
        st.title('MSA with GraphRAG')

    
def display_chat_history():
    """Display the chat history from session state"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            # Show relevant info for assistant messages
            if message["role"] == "assistant":
                if st.session_state.config['relevant_info'] == 'On':
                    if message.get('relevant_info') is not None:
                        Relevant_Information(message['relevant_info'])
# Define different background colors for alternating items
def display_retrieval_list(relevant_list:List[Tuple[str,str]]):
    bg_colors = [
        "rgba(255, 235, 238, 0.3)", # Light red
        "rgba(227, 242, 253, 0.3)", # Light blue  
        "rgba(232, 245, 233, 0.3)", # Light green
        "rgba(255, 243, 224, 0.3)", # Light orange
        "rgba(243, 229, 245, 0.3)"  # Light purple
    ]
    
    # Display each item with alternating background colors
    for i, item in enumerate(relevant_list):
        # Use modulo to cycle through colors
        bg_color = bg_colors[i % len(bg_colors)]
        
        # Add stronger border and increased opacity for better visibility
        st.markdown(
            f"""
            <div style='
                background-color: {bg_color}; 
                padding: 12px;
                margin: 8px 0;
                border-radius: 8px;
                border: 1px solid rgba(255,255,255,0.1);
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                position: relative;
            '>
                <div style='
                    position: absolute;
                    top: 4px;
                    left: 8px;
                    font-size: 0.8em;
                    color: rgba(0,0,0,0.6);
                '>
                    {item[1]}
                </div>
                <div style='
                    margin-top: 16px;
                '>
                    {item[0]}
                </div>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
def display_image_gallery(image_infos, title: str = "📷 Associated Images", max_images: int = 9):
    """Render a simple grid gallery for associated images if any."""
    if not image_infos:
        return
    st.markdown(title)
    cols = st.columns(3)
    shown = 0
    for i, img in enumerate(image_infos):
        if shown >= max_images:
            break
        path = img.get('path')
        caption = ", ".join(img.get('entities', [])) if img.get('entities') else None
        if path and os.path.exists(path):
            with cols[i % 3]:
                st.image(path, use_container_width=True, caption=caption)
                shown += 1
                        
def add_message(role: str, user_input: str):
    """Add a message to the chat history"""
    
    with st.chat_message(role):
        
        # Only show Relevant_Information for assistant messages
        if role == "assistant":
            # Create placeholders for status and message
            status_placeholder = st.empty()
            message_placeholder = st.empty()
            full_response = ""
            # Show retrieval status
            with status_placeholder.status("Retrieving relevant information..."):
               
                
                searched = st.session_state.settings['search_engine'].search(user_input)

                # Generate and store relevant info
                if st.session_state.settings['relevant_info'] == 'On':
                    if searched.retrieved_list is not None:
                        display_retrieval_list(searched.retrieved_list)
                        # Show associated images if any were linked
                        try:
                            if getattr(searched, 'associated_images', None):
                                display_image_gallery(searched.associated_images)
                        except Exception as e:
                            # Don't block the chat if images fail to render
                            st.caption(f"(Image display skipped: {e})")
            
            # Show generation status
            with status_placeholder.status("Generating response..."):
                content = st.session_state.settings['search_engine'].stream_answer(user_input,searched.structured_prompt)
                for chunk in content:
                    for char in chunk:
                        full_response += char
                        message_placeholder.markdown(full_response + "▌")
                        time.sleep(0.02)  # Simulate typing delay
                message_placeholder.markdown(full_response)
            if st.session_state.config['relevant_info'] == 'On':
                st.session_state.messages.append({"role": "assistant", "content":full_response , "relevant_info":searched.retrieved_list})
            else:
                st.session_state.messages.append({"role": "assistant", "content":full_response})
            
            with status_placeholder.status("✅ Retrieval and generation completed"):
                pass
                
            
        else:
            st.write(user_input)
            
        st.session_state.messages.append({"role": role, "content": user_input})
    
def handle_user_input():
    """Handle user input and generate assistant response"""
    if user_input := st.chat_input('Enter your text here:'):
        # Handle user message
        add_message("human", user_input)
        
        # Get and handle assistant response
        add_message("assistant", user_input)
        
def check_building_status(placeholder):
    """Check the building status"""
    with placeholder.status("Checking Building Status"): 
            if os.path.exists(os.path.join(st.session_state.config['main_folder'], 'info/state.json')):
                with open(os.path.join(st.session_state.config['main_folder'], 'info/state.json'), 'r') as f:
                    state = json.load(f)
                st.markdown(f"🔄 Building Status: {state['Current_state']}")
                return True
            else:
                st.markdown("🔹 Building Status: Not Built")
                return False


def sidebar():
    """Display the left sidebar with user input and assistant response"""
    with st.sidebar:
        st.title('Building Status')
        Build_Status = st.empty()
        Building = st.empty()
        check_building_status(Build_Status)
        
            
        if st.button("Build/Update",key="start_building"):
            state_observer = State_Observer(Building)
            state_observer.reset(total_tasks=["1. Document Processing", "2. Text Processing", "3. Graph Processing", "4. Attribute Processing", "5. Embedding Processing", "6. Summary Processing", "7. HNSW Processing","8. Finished"],desc="Building the NodeRAG")
            state_controller = NG(NGConfig(all_config()),web_ui=True)
            state_controller.add_observer(state_observer)
            state_controller.run()
            state_observer.close()
            
        if check_building_status(Build_Status):
            Enable_Search = st.toggle("Search Engine", value=True)
            
            if Enable_Search and not st.session_state.settings.get('engine_running'):
                st.session_state.settings['search_engine'] = NGSearch(NGConfig(all_config()))
                st.session_state.settings['engine_running'] = True
                st.write("Search Engine is running")
                if not st.session_state.indices:
                    st.session_state.indices = json.load(open(os.path.join(st.session_state.config['main_folder'], 'info/indices.json'), 'r'))
                
            elif not Enable_Search and st.session_state.settings.get('engine_running'):
                st.session_state.settings['engine_running'] = False
                st.session_state.settings['search_engine'] = None
                st.write("Search Engine is stopping")

            else:
                pass
        
        if st.session_state.settings.get('engine_running'):
            with st.expander("📑 Available Indices", expanded=False):
                if st.session_state.indices:
                    for key, value in st.session_state.indices.items():
                        st.markdown(f"**{key.replace('_',' ')}**: {value}")
                else:
                    st.markdown("No indices available")
        
        st.title("Settings")
        
        # Get current working directory as default folder
        
        # RAG Build Settings
        with st.expander("🔧 RAG Settings", expanded=False):
            # Basic Settings
            st.markdown("Main Folder: " + st.session_state.config['main_folder'])
            
            new_folder = st.text_input("Enter folder path:",key="main_folder")
            
            if new_folder:
                if os.path.exists(new_folder):
                    new_folder = new_folder.strip().strip('"\'')
                    st.session_state.config['main_folder'] = new_folder
                else:
                    st.error("Invalid folder path")
            
            st.session_state.config['language'] = st.selectbox(
                "Language",
                ["English", "Chinese"],
                index=["English", "Chinese"].index(st.session_state.config['language']),
                help="Processing language"
            )
            
            st.session_state.config['docu_type'] = st.selectbox(
                "Document Type",
                ["mixed", "md", "txt", "docx", "pdf"],
                index=["mixed", "md", "txt", "docx", "pdf"].index(st.session_state.config.get('docu_type', 'mixed')),
                help="Type of documents to process"
            )

            # Chunking Settings
            st.session_state.config['chunk_size'] = st.slider(
                "Chunk Size", 
                min_value=800,
                max_value=2000,
                value= st.session_state.config['chunk_size'],
                step=50,
                help="Size of text chunks for processing"
            )
            
            st.session_state.config['embedding_batch_size'] = st.number_input(
                "Embedding Batch Size",
                min_value=1,
                max_value=100,
                value= st.session_state.config['embedding_batch_size'],
                step=5,
                help="Number of embeddings to process in one batch"
            )


            # HNSW Index Settings
            st.subheader("HNSW Index Settings")
        
            st.session_state.config['dim'] = st.number_input(
                "Dimension",
                min_value=256,
                max_value=2048,
                value= st.session_state.config['dim'],
                help="Dimension of the embedding"
            )
            st.session_state.config['m'] = st.number_input(
                "M Parameter",
                min_value=5,
                max_value=100,
                value= st.session_state.config['m'],
                help="HNSW M parameter (max number of connections per layer)"
            )
            st.session_state.config['ef'] = st.number_input(
                "EF Parameter",
                min_value=50,
                max_value=500,
                value= st.session_state.config['ef'],
                help="HNSW ef parameter (size of dynamic candidate list)"
            )
            st.session_state.config['m0'] = st.radio(
                "M0 Parameter",
                [None],
                index=0,
                help="HNSW m0 parameter (number of bi-directional links)"
            )

            # Summary and Search Settings
            st.subheader("Summary and Search Settings")
        
        
            st.session_state.config['Hcluster_size'] = st.slider(
                "Hcluster Size",
                min_value=39,
                max_value=80,
                value=st.session_state.config['Hcluster_size'],
                step=1,
                help="Size of High level elements cluster"
            )
                

            

        # Model and Embedding Settings
        
        with st.expander("🤖 Model & Embedding Settings", expanded=False):
            st.subheader("Model settings")
            st.session_state.model_config['service_provider'] = st.selectbox(
                "Service Provider",
                ["openai",'gemini'],
                index=["openai",'gemini'].index(st.session_state.model_config['service_provider']),
                help="AI service provider"
            )
            if st.session_state.model_config['service_provider'] == 'openai':
                st.session_state.model_config['model_name'] = st.selectbox(
                    "Language Model",
                    ["gpt-4o-mini","gpt-4o"],
                    help="Select the language model to use"
                )
            elif st.session_state.model_config['service_provider'] == 'gemini':
                st.session_state.model_config['model_name'] = st.selectbox(
                    "Language Model",
                    ["gemini-2.0-flash-lite-preview-02-05"],
                    help="Select the language model to use"
                )
            
            st.markdown(f'api_keys: {st.session_state.model_config["api_keys"][:10] + "..."}')
            model_keys = st.text_input("Enter API Key:",key="model_keys")
            if model_keys:
                st.session_state.model_config['api_keys'] = model_keys.strip().strip('"\'')
                
            st.session_state.model_config['temperature'] = st.slider(
                "Model Temperature",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.model_config['temperature']),
                step=0.05,
                help="Temperature for model generation"
            )
            st.session_state.model_config['max_tokens'] = st.number_input(
                "Model Max Tokens",
                min_value=500,
                max_value=10000,
                value=int(st.session_state.model_config['max_tokens']),
                step=100,
                help="Maximum number of tokens for model generation"
            )
            st.session_state.model_config['rate_limit'] = st.number_input(
                "API Rate Limit (requests at a time)",
                min_value=1,
                max_value=50,
                value=int(st.session_state.model_config['rate_limit']),
                step=1,
                help="Rate limit for API calls"
            )
            
            # Embedding settings
            st.subheader("Embedding settings")
            st.session_state.embedding_config['service_provider'] = st.selectbox(
                "Embedding Provider",
                ["openai_embedding","gemini_embedding"],
                index=["openai_embedding","gemini_embedding"].index(st.session_state.embedding_config['service_provider']),
                help="Embedding service provider"
            )
            if st.session_state.embedding_config['service_provider'] == 'openai_embedding':
                st.session_state.embedding_config['embedding_model_name'] = st.selectbox(
                    "Embedding Model",
                    ["text-embedding-3-small", "text-embedding-3-large"],
                    help="Model used for generating embeddings"
                )
            elif st.session_state.embedding_config['service_provider'] == 'gemini_embedding':
                st.session_state.embedding_config['embedding_model_name'] = st.selectbox(
                    "Embedding Model",
                    ["text-embedding-004"],
                    help="Model used for generating embeddings"
                )
            st.markdown(f'api_keys: {st.session_state.embedding_config["api_keys"][:10] + "..."}')
            embedding_keys = st.text_input("Enter API Key:",key="embedding_keys")
            if embedding_keys:
                st.session_state.embedding_config['api_keys'] = embedding_keys.strip().strip('"\'')
            
            # Rate limits
            st.session_state.embedding_config['rate_limit'] = st.number_input(
                "API Rate Limit (requests/second)",
                min_value=1,
                max_value=50,
                value=st.session_state.embedding_config['rate_limit'],
                step=1,
                help="Rate limit for API calls"
            )
            
                    
        
        
        # File upload menu
        with st.expander("📁 Document Upload", expanded=False):
            uploaded_files = st.file_uploader(
                "Upload your documents",
                accept_multiple_files=True,
                type=['txt', 'doc', 'docx', 'md', 'pdf']
            )
            
            base_folder = st.session_state.config['main_folder']
            input_folder = base_folder if os.path.basename(base_folder.rstrip('/\\')).lower() == 'input' else os.path.join(base_folder, 'input')
            if uploaded_files: 
                if show_confirmation_dialog(f"Are you sure you want to upload file to {input_folder}?"):
                    os.makedirs(input_folder, exist_ok=True)
                    for file in uploaded_files:
                        file_path = os.path.join(input_folder, file.name)
                        with open(file_path, 'wb') as f:
                            f.write(file.getbuffer())
                    st.write("Files uploaded successfully")
                
            if os.path.exists(input_folder):
                input_files = os.listdir(input_folder)
                if input_files:
                    st.markdown("### 📄 Files in Input Folder")
                    for file in input_files:
                        st.markdown(f"<div style='margin-left:20px;'><i>📝 {file}</i></div>", unsafe_allow_html=True)
                else:
                    st.write("Input folder is empty")
                    
        # Settings menu
        with st.expander("🔍 Search Settings", expanded=False):
            st.session_state.config['relevant_info'] = st.selectbox(
                'Relevant Information', 
                ['On', 'Off'],
                index=0 
            )
            # Search settings
            
            st.session_state.config['unbalance_adjust'] = st.checkbox(
                "Unbalance  Adjust",
                value=True,
                help="Whether to adjust for unbalanced data"
            )
        
            st.session_state.config['cross_node'] = st.number_input(
                "Cross Node Number",
                min_value=1,
                max_value=50,
                value=st.session_state.config['cross_node'],
                step=1,
                help="Number of cross node"
            )
            
            st.session_state.config['Enode'] = st.number_input(
                "Entity Node",
                min_value=1,
                max_value=50,
                value=st.session_state.config['Enode'],
                step=1,
                help="Number of entity node"
            )
            
            st.session_state.config['Rnode'] = st.number_input(
                "Relation Node",
                min_value=1,
                max_value=50,
                value=st.session_state.config['Rnode'],
                step=1,
                help="Number of relation node"
            )
            
            st.session_state.config['Hnode'] = st.number_input(
                "High Level Node",
                min_value=1,
                max_value=50,
                value=st.session_state.config['Hnode'],
                step=1,
                help="Number of high level node"
            )
            
            st.session_state.config['HNSW_results'] = st.number_input(
                "HNSW Results",
                min_value=1,
                max_value=50,
                value=st.session_state.config['HNSW_results'],
                step=1,
                help="Number of top results to return"
            )
            
            st.session_state.config['ppr_alpha'] = st.slider(
                "PPR Alpha",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.config['ppr_alpha'],
                step=0.01,
                help="Alpha for PPR"
            )
            st.session_state.config['ppr_max_iter'] = st.number_input(
                "PPR Max Iter",
                min_value=1,
                max_value=100,
                value=st.session_state.config['ppr_max_iter'],
                step=1,
                help="Maximum number of iterations for PPR"
            )
            st.session_state.config['similarity_weight'] = st.slider(
                "Similarity Weight",
                min_value=1.0,
                max_value=3.0,
                value=float(st.session_state.config['similarity_weight']),
                step=0.1,
                help="Weight for similarity"
            )
            st.session_state.config['accuracy_weight'] = st.slider(
                "Accuracy Weight",
                min_value=1.0,
                max_value=3.0,
                value=float(st.session_state.config['accuracy_weight']),
                step=0.5,
                help="Weight for accuracy"
            )
        # Save config button
        def _on_save_config_click():
            # Trigger reload; ignore return value to satisfy callback signature
            reload_search_engine()
        st.button("💾 Save Configuration", on_click=_on_save_config_click)
        
        # MMLongBench Evaluation Section
        with st.expander("📊 MMLongBench Benchmark", expanded=False):
            st.markdown("### Run MMLongBench Evaluation")
            
            mmlongbench_path = st.text_input(
                "MMLongBench Path:",
                value=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'MMLongBench'),
                help="Path to MMLongBench directory"
            )
            
            if st.button("🚀 Run Evaluation", key="run_mmlongbench"):
                if not st.session_state.settings.get('engine_running'):
                    st.error("Please enable the Search Engine first!")
                else:
                    try:
                        from NodeRAG.utils.mmlongbench_eval import load_mmlongbench_samples, run_mmlongbench_evaluation
                        
                        samples_path = os.path.join(mmlongbench_path, 'data', 'samples.json')
                        if not os.path.exists(samples_path):
                            st.error(f"Samples file not found at {samples_path}")
                        else:
                            with st.spinner("Loading MMLongBench samples..."):
                                samples = load_mmlongbench_samples(samples_path)
                            st.success(f"✓ Loaded {len(samples)} samples")
                            
                            st.info("⏳ Running evaluation... Progress will be shown in the terminal. This will take a while!")
                            st.warning("💡 **Tip**: Check your terminal/console to see real-time progress")
                            
                            # Use a container to show status
                            status_container = st.empty()
                            
                            eval_output_dir = os.path.join(st.session_state.config['main_folder'], 'mmlongbench_results')
                            
                            # Run evaluation with progress updates
                            with status_container.container():
                                st.write("🔄 Evaluation in progress...")
                                st.write("Check terminal for detailed progress")
                            
                            results = run_mmlongbench_evaluation(
                                st.session_state.settings['search_engine'],
                                samples,
                                eval_output_dir
                            )
                            
                            # Clear status
                            status_container.empty()
                            
                            st.success("✅ Evaluation completed!")
                            
                            # Display results in columns
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Overall Accuracy", f"{results['accuracy']:.4f}")
                            with col2:
                                st.metric("Overall F1 Score", f"{results['f1']:.4f}")
                            with col3:
                                st.metric("Total Samples", results['total_samples'])
                            
                            if 'results_path' in results:
                                st.info(f"📁 Results saved to: {results['results_path']}")
                            if 'report_path' in results:
                                st.info(f"📄 Report saved to: {results['report_path']}")
                                
                            # Show report if available
                            if 'report_path' in results and os.path.exists(results['report_path']):
                                with st.expander("📊 View Detailed Report"):
                                    with open(results['report_path'], 'r') as f:
                                        report_content = f.read()
                                    st.text_area("Evaluation Report", report_content, height=300)
                    except Exception as e:
                        st.error(f"❌ Error running evaluation: {str(e)}")
                        import traceback
                        with st.expander("🔍 View Error Details"):
                            st.code(traceback.format_exc())
            
            
            

           
def Relevant_Information(relevant_list):
    """Display relevant information for a specific message or the latest one"""
    with st.expander("Relevant Information"):
       display_retrieval_list(relevant_list)
       
       
def show_confirmation_dialog(message):
    """Show a confirmation dialog"""
    dialog = st.empty()
    with dialog.container():
        st.write(message)
        col1, col2 = st.columns([1, 1])
        
        if col1.button("yes", key="confirm"):
            dialog.empty()  
            return True
        if col2.button("no", key="cancel"):
            dialog.empty() 
            return False
        
def reload_search_engine():
    """Reload the search engine with current config"""
    save_config(st.session_state.web_ui_config_path)
    if st.session_state.settings.get('engine_running'):
        st.session_state.settings['search_engine'] = NGSearch(NGConfig(all_config()))
        if st.session_state.main_folder != st.session_state.config['main_folder']:
            st.session_state.main_folder = st.session_state.config['main_folder']
            st.session_state.indices = json.load(open(os.path.join(st.session_state.config['main_folder'], 'info/indices.json'), 'r'))
            # Config files should be in parent directory if main_folder ends with 'input'
            main_folder = st.session_state.config['main_folder']
            if os.path.basename(main_folder.rstrip('/\\')).lower() == 'input':
                config_base = os.path.dirname(main_folder)
            else:
                config_base = main_folder
            st.session_state.original_config_path = os.path.join(config_base, 'Node_config.yaml')
            st.session_state.web_ui_config_path = os.path.join(config_base, 'web_ui_config.yaml')
        return True
    return False


# Main chat interface
DEFAULT_MAIN_FOLDER = 'data'
if os.path.exists(st.session_state.web_ui_config_path):
    load_config(st.session_state.web_ui_config_path)
elif os.path.exists(st.session_state.original_config_path):
    load_config(st.session_state.original_config_path)
else:
    # Create config file with the correct main_folder
    NGConfig.create_config_file(config_base_folder)
    # Update the created config file to use the correct main_folder
    with open(st.session_state.original_config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    config_data['config']['main_folder'] = DEFAULT_MAIN_FOLDER
    with open(st.session_state.original_config_path, 'w') as f:
        yaml.dump(config_data, f)
    load_config(st.session_state.original_config_path)
display_header()
sidebar()

def graph_visualization():
    """Visualize the processed graph using networkx (simple layout)."""
    base_folder = st.session_state.config['main_folder']
    graph_path = os.path.join(base_folder, 'cache', 'graph.pkl')
    st.subheader("Graph visualization")
    if not os.path.exists(graph_path):
        st.info(f"Graph file not found at {graph_path}. Build the graph first from the sidebar.")
        return
    # Load the graph
    try:
        with open(graph_path, 'rb') as f:
            G = pickle.load(f)
    except Exception as e:
        st.error(f"Failed to load graph: {e}")
        return
    try:
        import networkx as nx  # type: ignore
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        st.info("Install 'networkx' and 'matplotlib' to view the graph.")
        return
    # Controls
    max_nodes = st.slider("Max nodes to display", min_value=50, max_value=2000, value=300, step=50)
    show_labels = st.checkbox("Show labels", value=False)
    show_entity_names = st.checkbox("Show entity/text on hover", value=True)
    # Subgraph selection by highest degree
    try:
        degrees = dict(G.degree())
        top_nodes = sorted(degrees.keys(), key=lambda n: degrees[n], reverse=True)[:max_nodes]
        H = G.subgraph(top_nodes).copy()
    except Exception:
        # Fallback to slicing if degree not available
        nodes = list(G.nodes())[:max_nodes]
        H = G.subgraph(nodes).copy()
    # Optionally load id->text mapping for hover tooltips
    id_to_text = {}
    if show_entity_names:
        try:
            cache_dir = os.path.join(base_folder, 'cache')
            mapping_candidates = [
                os.path.join(cache_dir, 'semantic_units.parquet'),
                os.path.join(cache_dir, 'entities.parquet'),
                os.path.join(cache_dir, 'relationship.parquet'),
                os.path.join(cache_dir, 'attributes.parquet'),
                os.path.join(cache_dir, 'high_level_elements.parquet'),
                os.path.join(cache_dir, 'text.parquet'),
                os.path.join(cache_dir, 'high_level_elements_titles.parquet'),
            ]
            mapping_list = [p for p in mapping_candidates if os.path.exists(p)]
            if mapping_list:
                from NodeRAG.storage.graph_mapping import Mapper
                mapper = Mapper(mapping_list)
                # returns (id_to_text, accurate_id_to_text, relationships?) in this codebase
                gen_result = mapper.generate_id_to_text(['entity','high_level_element_title'])
                if isinstance(gen_result, tuple) and len(gen_result) >= 1:
                    id_to_text = gen_result[0] or {}
        except Exception:
            id_to_text = {}

    # Color by node type
    type_colors = {
        'entity': '#1f77b4',
        'relationship': '#ff7f0e',
        'high_level_element_title': '#2ca02c',
        'attribute': '#9467bd',
    }
    def node_color(n):
        return type_colors.get(H.nodes[n].get('type', 'other'), '#aaaaaa')

    # Try interactive PyVis first (zoom/pan)
    try:
        from pyvis.network import Network  # type: ignore
        import streamlit.components.v1 as components
        net = Network(height="750px", width="100%", bgcolor="#ffffff")
        net.barnes_hut()
        # Add nodes
        for n in H.nodes():
            title = id_to_text.get(n, str(n)) if show_entity_names else None
            net.add_node(str(n), label=(str(n) if show_labels else ""), color=node_color(n), title=title)
        # Add edges
        for u, v in H.edges():
            net.add_edge(str(u), str(v))
        # Enable zoom and drag interactions
        net.set_options('''
        var options = {
          "interaction": {"zoomView": true, "dragView": true},
          "physics": {"stabilization": true}
        };
        ''')
        # Render to HTML and embed
        import tempfile
        import os as _os
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
            try:
                net.write_html(tmp.name)
            except Exception:
                net.show(tmp.name)
            html_str = open(tmp.name, 'r', encoding='utf-8').read()
        try:
            _os.unlink(tmp.name)
        except Exception:
            pass
        components.html(html_str, height=780, scrolling=True)
        return
    except Exception:
        pass

    # Fallback to Plotly (also interactive with zoom/pan)
    try:
        import plotly.graph_objects as go  # type: ignore
        import networkx as nx  # type: ignore
        try:
            pos = nx.spring_layout(H, seed=42)
        except Exception:
            pos = nx.random_layout(H)
        # Edges
        edge_x = []
        edge_y = []
        for u, v in H.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), mode='lines', hoverinfo='none')
        # Nodes
        node_x = []
        node_y = []
        node_text = []
        node_hover = []
        node_color_vals = []
        for n in H.nodes():
            x, y = pos[n]
            node_x.append(x)
            node_y.append(y)
            node_text.append(str(n))
            node_hover.append(id_to_text.get(n, str(n)) if show_entity_names else str(n))
            node_color_vals.append(node_color(n))
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text' if show_labels else 'markers',
            text=node_text if show_labels else None,
            textposition='top center',
            marker=dict(size=8, color=node_color_vals),
            hovertext=node_hover,
            hoverinfo='text'
        )
        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(showlegend=False, hovermode='closest', margin=dict(b=10, l=10, r=10, t=10))
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        st.plotly_chart(fig, use_container_width=True)
        return
    except Exception:
        pass

    # Final fallback: static matplotlib
    try:
        import networkx as nx  # type: ignore
        import matplotlib.pyplot as plt  # type: ignore
        try:
            pos = nx.spring_layout(H, seed=42)
        except Exception:
            pos = nx.random_layout(H)
        fig, ax = plt.subplots(figsize=(10, 8))
        nx.draw_networkx_edges(H, pos, ax=ax, alpha=0.25)
        nx.draw_networkx_nodes(H, pos, node_color=[node_color(n) for n in H.nodes()], node_size=50, ax=ax)
        if show_labels:
            labels = {n: str(n) for n in H.nodes()}
            nx.draw_networkx_labels(H, pos, labels=labels, font_size=6, ax=ax)
        ax.axis('off')
        st.pyplot(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Graph rendering failed: {e}")

tab_chat, tab_graph = st.tabs(["💬 Chat", "🕸️ Graph"])
with tab_chat:
    display_chat_history()
    handle_user_input()
with tab_graph:
    graph_visualization()
