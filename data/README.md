# Data Directory

This directory stores all input documents and generated outputs.

## Structure

```
data/
├── input/              # Place your PDF files here
├── cache/              # Generated during build (embeddings, graphs, etc.)
├── info/               # Build metadata and state
├── extracted_images/   # Images extracted from PDFs
└── entity_image_mappings/  # Entity-to-image mapping files
```

## Usage

1. **Add Documents**: Place your PDF files in `data/input/`
2. **Build**: Click "Build/Update" in the Streamlit UI or run the build pipeline
3. **Query**: Enable "Search Engine" and ask questions

## Note

- All paths in the system use relative references from the project root
- Outputs are stored alongside `input/` in the `data/` directory
- You can change the main folder in the Streamlit sidebar settings
