import os
import subprocess
import glob
import re
import yaml
from tqdm import tqdm

def create_dialogue_kg(line, dialogue_name):
    """
    Process a dialogue line with GraphRAG
    
    Args:
        line (str): The dialogue line to process
        dialogue_name (str): The name of the dialogue (e.g., "train_step0")
    """
    # Write the dialogue to the input file
    line = line.strip()
    # with open(r'C:\Users\Aarushi\Desktop\FYP\generate_dialogue_kgs\UniCRS_GraphRAG\Recommendation_GraphRAG\input\current_line.txt', 'w') as current_file:
    #     current_file.write(line)
    with open(r'/home/Nema/UniCRS_GraphRAG/Recommendation_GraphRAG/input/current_line.txt', 'w', encoding='utf-8') as current_file:
        current_file.write(line)


    # Set environment variable
    env = os.environ.copy()  # Copy existing environment variables
    env["FILE_NAME"] = dialogue_name  # Set FILE_NAME dynamically
    
    
    # Run the graphrag command with the dialogue name
    process = subprocess.run(
    ['graphrag', 'index', '--root', r'/home/Nema/UniCRS_GraphRAG/Recommendation_GraphRAG'],
    capture_output=True,
    text=True,
    env=env,
    encoding='utf-8',  # Force UTF-8 encoding
    errors='replace'    # Replace unsupported characters
)

    
    success = process.returncode == 0
    message = "Command completed successfully" if success else f"Error: {process.stderr}"
    
    return success, message

    
def extract_dialogue_name(file_path):
    """Extract the dialogue name (e.g., train_step0) from the file path"""
    match = re.search(r'(train_step\d+)', file_path)
    if match:
        return match.group(1)
    # Fallback to just the filename without extension
    return os.path.splitext(os.path.basename(file_path))[0]

def main():
    # Get all training dialogue files
    dialogue_files = glob.glob(r"/home/Nema/UniCRS_GraphRAG/Recommendation_GraphRAG/dialogue_outputs/train/train_step*.txt")

    # Sort numerically
    dialogue_files.sort(key=lambda x: int(re.search(r'train_step(\d+)', x).group(1)))

    # Limit to first 1000 files
    dialogue_files = dialogue_files[1000:1425]
    
    print(f"Processing {len(dialogue_files)} dialogue files...")
    
    # Process each file
    success_count = 0
    failure_count = 0
    
    
    for i, file_path in enumerate(tqdm(dialogue_files)):
        try:
            # Extract dialogue name
            dialogue_name = extract_dialogue_name(file_path)
            print(dialogue_name)
            
            # Read the dialogue from the file
            with open(file_path, 'r', encoding='utf-8') as f:
                dialogue = f.read()
            
            # Process the dialogue with its name
            success, message = create_dialogue_kg(dialogue, dialogue_name)

            # print(message)
            
            # Update counts and log
            if success:
                success_count += 1
            else:
                failure_count += 1
            
        except Exception as e:
            # Log any exceptions
            failure_count += 1
    # Print summary
    print(f"Processing complete!")
    print(f"Successful: {success_count}")
    print(f"Failed: {failure_count}")
    # print(f"Detailed logs saved to dialogue_kg_logs/processing_results.txt")

if __name__ == "__main__":
    main()