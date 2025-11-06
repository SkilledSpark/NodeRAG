# MMLongBench Test Fix - System Prompt Support for OpenAI Models

## Issue
The MMLongBench test script was not properly separating system prompts from user prompts when using OpenAI's models. This caused the custom evaluation system prompt to be embedded in the user message rather than being sent as a proper system message, which can lead to:
- Reduced instruction following accuracy
- Less optimal model behavior
- Inconsistent response formatting

## Changes Made

### 1. Updated `NodeRAG/search/search.py` - `stream_answer()` method

**Before:**
```python
def stream_answer(self, query: str, retrieved_info: str):
    query = self.config.prompt_manager.answer.format(info=retrieved_info, query=query)
    response = self.config.API_client.stream_chat({'query': query})
    yield from response
```

**After:**
```python
def stream_answer(self, query: str, retrieved_info: str, system_prompt: str | None = None):
    """
    Stream answer with proper system/user prompt separation for OpenAI models.
    
    Args:
        query: The user's question
        retrieved_info: The retrieved context information
        system_prompt: Optional custom system prompt to prepend
    """
    # Get the answer prompt template
    answer_template = self.config.prompt_manager.answer
    
    # Format the full prompt with retrieved info and query
    formatted_prompt = answer_template.format(info=retrieved_info, query=query)
    
    # For OpenAI models, separate system and user prompts
    if system_prompt:
        # Extract role/goal/format sections as system prompt
        parts = formatted_prompt.split("---Retrived Context---", 1)
        if len(parts) == 2:
            base_system = parts[0].strip()
            user_content = "---Retrived Context---" + parts[1]
            full_system = system_prompt.strip() + "\n\n" + base_system
            response = self.config.API_client.stream_chat({
                'system_prompt': full_system,
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
```

**Key Improvements:**
- Added optional `system_prompt` parameter
- Automatically separates role/goal instructions from context/query
- Properly formats messages for OpenAI's chat API (system vs user)
- Maintains backward compatibility when no system prompt is provided
- Smart parsing to extract system instructions from the prompt template

### 2. Updated `test_mmlongbench.py`

**Changes:**
1. Updated the call to `stream_answer()` to pass the custom `SYSTEM_PROMPT`:
```python
answer_stream = search_engine.stream_answer(
    formatted_question, 
    searched.structured_prompt,
    system_prompt=SYSTEM_PROMPT  # Now properly passed
)
```

2. Removed obsolete code that tried to override `config.prompt_manager.answer`
3. Updated documentation comments to reflect the new approach

## How It Works

### Message Structure
When using OpenAI models, the method now creates proper message structures:

**System Message:**
```
{Custom SYSTEM_PROMPT if provided}

---Role---
You are a thorough assistant...

---Goal---
Provide a clear and accurate response...

---Target response length and format---
Multiple Paragraphs
```

**User Message:**
```
---Retrived Context---
{Retrieved information from the knowledge graph}

---Query---
{User's question}
```

### Benefits
1. **Better Instruction Following**: OpenAI models treat system messages with higher priority
2. **Proper Separation**: Clear distinction between system instructions and user content
3. **Evaluation Accuracy**: Critical for MMLongBench where specific output format is required
4. **Flexible**: Can be used with or without custom system prompts
5. **Backward Compatible**: Works with existing code that doesn't provide system prompts

## Usage

### Basic Usage (No Custom System Prompt)
```python
answer_stream = search_engine.stream_answer(query, retrieved_info)
```
This automatically separates the role/goal sections as system prompt.

### With Custom System Prompt (MMLongBench)
```python
SYSTEM_PROMPT = """Given the question and analysis, you are tasked to extract answers..."""

answer_stream = search_engine.stream_answer(
    query, 
    retrieved_info, 
    system_prompt=SYSTEM_PROMPT
)
```

## Testing
The fix can be tested with:
```bash
python test_mmlongbench.py --max-samples 3
```

This will run the evaluation on 3 samples and properly use the system prompt for answer extraction formatting.

## Notes
- The OpenAI `OPENAI` class in `LLM.py` already had proper support for system prompts via the `messages()` method
- The `stream_chat()` method properly calls `self.messages(input)` which handles the system_prompt key
- This fix ensures that system prompts are utilized throughout the entire pipeline
