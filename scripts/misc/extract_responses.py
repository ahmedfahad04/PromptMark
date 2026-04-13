import json
import os
import re

# Load the sweet-mbpp-generation.json file
with open('dataset/sweet-mbpp-generation.json', 'r') as f:
    sweet_data = json.load(f)

# Load the sanitized-mbpp.json file
with open('dataset/sanitized-mbpp.json', 'r') as f:
    mbpp_data = json.load(f)

# Create a mapping of prompts to task_ids from sanitized-mbpp.json
prompt_to_task_id = {}
for item in mbpp_data:
    prompt = item['prompt']
    task_id = item['task_id']
    prompt_to_task_id[prompt] = task_id

# Create output directory
output_dir = 'extracted_responses'
os.makedirs(output_dir, exist_ok=True)

# Function to extract only code from a response
def extract_code_only(response):
    """
    Extract only the code part from response.
    The response contains multiple sections separated by triple quotes.
    Code sections are between triple quotes.
    """
    # Split by triple quotes
    parts = response.split('"""')
    
    # Code parts are at even indices (0, 2, 4, etc.) after the first split
    # But we need to find the actual code, not the prompts
    # The pattern is: """prompt"""\ncode\n"""prompt"""\ncode...
    
    # Let's extract all non-empty parts that don't look like prompts
    code_blocks = []
    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
        # Skip parts that look like prompts (contain "Write a" or "Your code should")
        if i % 2 == 1:  # Odd indices are prompts
            continue
        # Even indices (except 0) are code
        if i > 0 and part:
            # This is a code block
            # Extract only the first code block (the actual solution)
            lines = part.split('\n')
            code_lines = []
            for line in lines:
                # Stop at the next prompt
                if line.strip().startswith('"""') or 'Write a' in line or 'Your code should' in line:
                    break
                code_lines.append(line)
            if code_lines:
                return '\n'.join(code_lines).strip()
    
    # If we couldn't extract code properly, try a different approach
    # Look for Python code patterns (def, import, class, etc.)
    for i, part in enumerate(parts):
        part = part.strip()
        if any(part.startswith(keyword) for keyword in ['def ', 'import ', 'class ', 'from ']):
            # Found code, extract until next prompt
            lines = part.split('\n')
            code_lines = []
            for line in lines:
                if '"""' in line and code_lines:  # Stop at next docstring
                    break
                code_lines.append(line)
            return '\n'.join(code_lines).strip()
    
    return response.strip()

# Process each item in sweet-mbpp-generation.json
matched_count = 0
unmatched_count = 0
task_ids_found = set()

for idx, item in enumerate(sweet_data):
    if len(item) < 2:
        continue
    
    # Extract question (first element) and responses
    question = item[0]
    responses = item[1:5] if len(item) >= 5 else item[1:]
    
    # Extract the first prompt from the question
    parts = question.split('"""')
    if len(parts) >= 2:
        first_prompt = parts[1].strip()
        
        # Clean up the prompt for better matching
        # Remove test assertions and extra whitespace
        first_prompt_clean = first_prompt.split('Your code should')[0].strip()
        first_prompt_clean = first_prompt_clean.split('assert ')[0].strip()
        
        # Find matching task_id by comparing prompts
        task_id = None
        best_match_score = 0
        
        for prompt, tid in prompt_to_task_id.items():
            # Calculate similarity (simple word overlap)
            words1 = set(first_prompt_clean.lower().split())
            words2 = set(prompt.lower().split())
            
            if len(words1) == 0 or len(words2) == 0:
                continue
                
            overlap = len(words1 & words2)
            score = overlap / max(len(words1), len(words2))
            
            # If high similarity, consider it a match
            if score > 0.7 and score > best_match_score:
                best_match_score = score
                task_id = tid
        
        if task_id is not None and task_id not in task_ids_found:
            # Extract only the code from the first response
            if len(responses) > 0:
                code_only = extract_code_only(responses[0])
                
                # Save the code to a file
                filename = f"{output_dir}/{task_id}.py"
                with open(filename, 'w') as f:
                    f.write(code_only)
                
                task_ids_found.add(task_id)
                matched_count += 1
                print(f"Processed item {idx}: Matched to task_id {task_id}")
        elif task_id is None:
            unmatched_count += 1
            if idx < 10:  # Print first few for debugging
                print(f"Item {idx}: Could not find matching task_id for prompt: {first_prompt_clean[:80]}...")

print(f"\nSummary:")
print(f"Matched: {matched_count}")
print(f"Unmatched: {unmatched_count}")
print(f"Unique task_ids found: {len(task_ids_found)}")
print(f"Files saved in '{output_dir}/' directory")
