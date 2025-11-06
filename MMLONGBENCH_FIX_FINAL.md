# MMLongBench Test Fix - Two-Step Answer Generation

## Problem Identified

The original MMLongBench test was trying to do answer extraction in a single step, which caused the model to output the extraction format markers (`Extracted answer:`, `Answer format:`) directly instead of providing a clean answer. This resulted in 0.0 accuracy on all tests.

## Root Cause

The MMLongBench evaluation requires a **TWO-STEP** process:

1. **Step 1 - Analysis Generation**: Use the GraphRAG system to generate a detailed analysis based on retrieved context
2. **Step 2 - Answer Extraction**: Extract the final answer from the analysis in the required format

The original implementation was trying to combine both steps, which confused the model and led to incorrect outputs.

## Solution Implemented

### 1. Fixed `NodeRAG/search/search.py` - `stream_answer()` method

Updated to properly handle custom system prompts by **replacing** (not prepending) the default system prompt when a custom one is provided:

```python
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
        # No custom system prompt - use default behavior
        parts = formatted_prompt.split("---Retrived Context---", 1)
        if len(parts) == 2:
            system_part = parts[0].strip()
            user_part = "---Retrived Context---" + parts[1]
            response = self.config.API_client.stream_chat({
                'system_prompt': system_part,
                'query': user_part
            })
        else:
            response = self.config.API_client.stream_chat({'query': formatted_prompt})
    
    yield from response
```

**Key Changes:**
- When `system_prompt` is provided, it completely **replaces** the default system instructions
- Only the retrieved context and query are sent in the user message
- This prevents mixing of different instruction sets

### 2. Updated `test_mmlongbench.py` - Two-Step Evaluation

Implemented the proper two-step process:

```python
# STEP 1: Search and generate analysis using GraphRAG
print("  [Step 1/2] Searching GraphRAG...")
searched = search_engine.search(question)

print("  [Step 1/2] Generating analysis...")
# Generate analysis without custom system prompt (use default NodeRAG prompt)
analysis_stream = search_engine.stream_answer(
    question, 
    searched.structured_prompt
)

# Collect the analysis
analysis_parts = []
for chunk in analysis_stream:
    analysis_parts.append(chunk)

analysis_text = ''.join(analysis_parts)

# STEP 2: Extract formatted answer from the analysis
print("  [Step 2/2] Extracting formatted answer...")

# Format the extraction prompt with question and analysis
extraction_user_prompt = f"""Question: {question}
Analysis: {analysis_text}"""

# Use the OpenAI client directly to extract the answer
extraction_response = search_engine.config.API_client.request({
    'system_prompt': EXTRACTION_SYSTEM_PROMPT,
    'query': extraction_user_prompt
})
```

**Key Changes:**
1. **Step 1** uses the default NodeRAG answer prompt to generate comprehensive analysis
2. **Step 2** uses the MMLongBench extraction prompt to format the answer properly
3. Results now store both `analysis` and `prediction` for debugging
4. Clear progress indicators for each step

### 3. Updated Extraction System Prompt

The `EXTRACTION_SYSTEM_PROMPT` is now properly formatted and includes clear examples for the model:

```python
EXTRACTION_SYSTEM_PROMPT = """Given the question and analysis, you are tasked to extract answers with required formats from the free-form analysis.
- Your extracted answers should be one of the following formats: (1) Integer, (2) Float, (3) String and (4) List. 
If you find the analysis the question can not be answered from the given documents, type "Not answerable". 
Exception: If the analysis only tells you that it can not read/understand the images or documents, type "Fail to answer".
- Please make your response as concise as possible. Also note that your response should be formatted as below:
```
Extracted answer: [answer]
Answer format: [answer format]
```

[Examples included...]
"""
```

## How It Works Now

### Message Flow for Step 1 (Analysis Generation)

**System Message:**
```
---Role---
You are a thorough assistant responding to questions based on retrieved information.

---Goal---
Provide a clear and accurate response...
```

**User Message:**
```
---Retrived Context---
[GraphRAG retrieved context]

---Query---
[User's question]
```

**Output:** Detailed analysis of the question based on the context

### Message Flow for Step 2 (Answer Extraction)

**System Message:**
```
Given the question and analysis, you are tasked to extract answers with required formats...
[Examples of proper extraction format]
```

**User Message:**
```
Question: [Original question]
Analysis: [Generated analysis from Step 1]
```

**Output:** 
```
Extracted answer: [clean answer]
Answer format: [Integer/Float/String/List]
```

## Benefits

1. **Correct Separation of Concerns**: Analysis generation and answer extraction are separate operations
2. **Better Model Performance**: Each step has focused instructions without confusion
3. **Proper System Prompt Usage**: Custom prompts replace (not mix with) default prompts
4. **Debugging Support**: Both analysis and extracted answer are stored for inspection
5. **MMLongBench Compliance**: Output format matches expected structure

## Testing

Run the test with:
```bash
python test_mmlongbench.py --max-samples 3
```

Expected behavior:
- Step 1 generates comprehensive analysis from GraphRAG
- Step 2 extracts clean, formatted answers
- Results should have much higher accuracy scores
- Both analysis and prediction are saved in results

## Files Modified

1. `NodeRAG/search/search.py` - Updated `stream_answer()` to properly handle custom system prompts
2. `test_mmlongbench.py` - Implemented two-step evaluation process with proper prompt separation

## Previous Issues Fixed

- ❌ Model outputting format markers instead of answers
- ❌ System prompts being mixed with default prompts
- ❌ Single-step process trying to do two jobs at once
- ❌ 0.0 accuracy on all evaluation samples

## Expected Results

- ✅ Clean analysis generation from GraphRAG
- ✅ Proper answer extraction in required format
- ✅ Significantly improved accuracy scores
- ✅ Proper separation of system and user messages
- ✅ Both intermediate and final results available for debugging
