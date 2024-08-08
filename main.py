import os
import openai
import time
import random
import requests
import boto3
from dotenv import load_dotenv
from botocore.exceptions import NoCredentialsError, ClientError
from langchain.vectorstores import Qdrant
from langchain.embeddings.openai import OpenAIEmbeddings
import qdrant_client

# Load environment variables
script_dir = os.path.dirname(os.path.realpath(__file__))
env_path = os.path.join(script_dir, 'py.env')

load_dotenv(env_path)
folder_path = os.path.join(script_dir, "screenshots")

openai.api_key = os.getenv("OPENAI_API_KEY")
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_bucket_name = os.getenv("AWS_BUCKET_NAME")
embeddings = OpenAIEmbeddings()
collection_name = os.getenv("QDRANT_COLLECTION_NAME")

client = qdrant_client.QdrantClient(
    os.getenv("QDRANT_HOST"),
    api_key=os.getenv("QDRANT_API_KEY")
)

vector_store = Qdrant(
    client=client,
    collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
    embeddings=embeddings
)

s3_client = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

# Get the prompts function
def load_prompt(file_name):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    prompt_path = os.path.join(dir_path, f'{file_name}.txt')
    
    try:
        with open(prompt_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except FileNotFoundError:
        print(f"Could not find file: {prompt_path}")
        raise

# Extract prompts
vision_prompt = load_prompt('p_vision_prompt')
overarching_concepts = load_prompt('p_overarching_concepts')
vector_summarizer_prompt = load_prompt('p_vector_summarizer')

def call_gpt4_vision(image_url, vision_prompt):
    print('API request sent, waiting for response...')
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            temperature=0.1,
            messages=[
                {"role": "system", "content": vision_prompt},
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_url}}]},
            ],
            max_tokens=999,
        )
        # Try to extract the content from the response
        if response.choices:
            return response.choices[0].message.content
        else:
            print("No response content.")
            return "No response content."
    except Exception as e:
        # If there's an error (like unexpected response format), show the full response for debugging
        print(f"Error parsing response: {e}\nFull response: {e.response if hasattr(e, 'response') else 'No response attribute'}")
        return f"Error: {e}"
    
def call_gpt4_get_topics(content, overarching_concepts):
    print('Getting topics, waiting for response...')
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            messages=[
                {"role": "system", "content": overarching_concepts},
                {"role": "user", "content": "Vectores: " + "[" + content + "]"},
            ],
            max_tokens=999,
        )
        if response.choices:
            return response.choices[0].message.content
        else:
            print("No response content.")
            return "No response content."
    except Exception as e:
        print(f"Error parsing response: {e}\nFull response: {e.response if hasattr(e, 'response') else 'No response attribute'}")
        return f"Error: {e}"

def call_gpt4_vector_summarizer(topic, content, vector_summarizer_prompt, remaining_topics):
    if remaining_topics:
        remaining_topics_str = "; ".join(remaining_topics)
        remaining_topics_info = ". Lista de otros temas que se van a resumir (evita en la medida de lo posible abordarlos para no repetir contenido): " + remaining_topics_str + "."
    else:
        remaining_topics_info = ""

    formatted_prompt = vector_summarizer_prompt.format(topic=topic)
    print('API request sent for vector summarization, waiting for response...')
    print("Tema: " + topic + remaining_topics_info)

    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            temperature=0.1,
            messages=[
                {"role": "system", "content": formatted_prompt},
                {"role": "user", "content": "Tema: " + topic + remaining_topics_info + "\n\n" + "Vectores: " + "[" + content + "]"}
            ],
            max_tokens=999,
        )
        if response.choices:
            return response.choices[0].message.content
        else:
            print("No response content.")
            return "No response content."
    except Exception as e:
        print(f"Error parsing response: {e}\nFull response: {e.response if hasattr(e, 'response') else 'No response attribute'}")
        return f"Error: {e}"


def upload_to_s3(image_path, bucket_name, max_retries=3):
    attempts = 0
    while attempts < max_retries:
        attempts += 1
        try:
            # Upload file to S3 without specifying ACL
            s3_client.upload_file(image_path, bucket_name, os.path.basename(image_path))
            
            # Generate the URL for the uploaded file
            image_url = f"https://{bucket_name}.s3.amazonaws.com/{os.path.basename(image_path)}"
            print(f"Uploaded to S3 successfully: {image_url}")
            return image_url
        except (NoCredentialsError, ClientError) as e:
            print(f"Error uploading to S3 (Attempt {attempts}/{max_retries}): {e}")
        
        time.sleep(2)  # Wait before retrying
    
    return None

def verify_url_accessibility(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            print(f"URL is accessible")
            return True
        else:
            print(f"URL is not accessible, status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error accessing URL: {e}")
        return False

def process_images_in_folder(folder_path, vision_prompt, transcription_file_path, topic_name):
    images_processed = 0
    failed_uploads = []
    failed_transcriptions = []

    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp']  # Supported image extensions

    for filename in os.listdir(folder_path):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            images_processed += 1
            print(f"Processing {filename} from {os.path.basename(folder_path)} -> {topic_name}")
            image_path = os.path.join(folder_path, filename)
            
            image_url = upload_to_s3(image_path, aws_bucket_name)
            if not image_url:
                print(f"Failed to upload {filename} to S3 after multiple attempts.")
                failed_uploads.append(filename)
                continue
            
            # Verify the image URL is accessible
            if not verify_url_accessibility(image_url):
                failed_uploads.append(filename)
                continue
            
            print(f"Calling GPT-4 Vision with image URL")
            text = call_gpt4_vision(image_url, vision_prompt)
            if "Error:" in text:
                print(f"Failed to process {filename} with GPT-4 Vision.")
                failed_transcriptions.append(filename)
            else:
                print(f"Received transcription for {filename}")  # Show first 100 characters of the response
                with open(transcription_file_path, 'a', encoding='utf-8') as f:
                    f.write(text + '\n' + '-'*40 + '\n')
                
                # Delete the image from local storage
                try:
                    os.remove(image_path)
                    print(f"Deleted local image: {filename}")
                except Exception as e:
                    print(f"Failed to delete local image: {filename}. Error: {e}")
            
            print("--------------")
            
            waiting_time = random.randint(0, 2)
            print(f"Waiting {waiting_time} seconds to avoid rate limit...")
            print("--------------")
            time.sleep(waiting_time)

        else:
            print(f"Skipped non-image file: {filename}")
            print("--------------")

    if failed_uploads:
        print(f"Failed to upload the following images to S3: {failed_uploads}")
    if failed_transcriptions:
        print(f"Failed to transcribe the following images: {failed_transcriptions}")

def process_folder_recursively(folder_path, vision_prompt):
    print("Starting to process the folder recursively...")
    for root, dirs, files in os.walk(folder_path):
        topic_name = os.path.basename(root)
        for subdir in dirs:
            subdir_path = os.path.join(root, subdir)
            print(f"Processing subfolder: {subdir_path}")
            transcription_file_path = os.path.join(subdir_path, 'transcriptions.txt')
            process_images_in_folder(subdir_path, vision_prompt, transcription_file_path, topic_name)
    print("Finished processing the folder recursively!")

def count_vectors_in_file(file_path):
    if not os.path.exists(file_path):
        return 0
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return len([v for v in content.split('----------------------------------------') if v.strip()])

def count_vectors_in_main_folders(folder_path):
    print()
    print("Counting vectors in each main folder...")
    print("------------------------------")
    for root, dirs, files in os.walk(folder_path):
        if root == folder_path:
            for dir in dirs:
                main_folder_path = os.path.join(root, dir)
                total_vectors = 0
                for subdir_root, subdirs, subfiles in os.walk(main_folder_path):
                    for subdir in subdirs:
                        subdir_path = os.path.join(subdir_root, subdir)
                        transcription_file_path = os.path.join(subdir_path, 'transcriptions.txt')
                        if os.path.exists(transcription_file_path):
                            total_vectors += count_vectors_in_file(transcription_file_path)
                print(f"{dir}: {total_vectors} vectors")
            break  # Only process the first level of directories

    print("------------------------------")



def update_transcription_file_names(folder_path):
    print("Updating transcription file names to include the number of vectors...")
    for root, dirs, files in os.walk(folder_path):
        for subdir in dirs:
            subdir_path = os.path.join(root, subdir)
            transcription_file_path = os.path.join(subdir_path, 'transcriptions.txt')
            if os.path.exists(transcription_file_path):
                num_vectors = count_vectors_in_file(transcription_file_path)
                new_transcription_file_path = transcription_file_path.replace('transcriptions.txt', f'transcriptions_{num_vectors}.txt')
                os.rename(transcription_file_path, new_transcription_file_path)
                print(f"Renamed {transcription_file_path} to {new_transcription_file_path}")
            else:
                print(f"No transcription file found in {subdir_path}")

# Gets overarching concepts and makes individual vectors for each of them
def summarize_transcriptions(folder_path):
    print("Summarizing transcriptions...")
    errors = []

    main_folder = display_main_folders(folder_path)
    if not main_folder:
        return
    
    main_folder_path = os.path.join(folder_path, main_folder)
    
    sub_folder = display_sub_folders(main_folder_path)
    if not sub_folder:
        return
    
    sub_folder_path = os.path.join(main_folder_path, sub_folder)
    process_sub_folder(sub_folder_path, errors)

    # Process the remaining subfolders in sequence
    remaining_sub_folders = [f for f in os.listdir(main_folder_path) if os.path.isdir(os.path.join(main_folder_path, f)) and f != sub_folder]
    
    for sub_folder in remaining_sub_folders:
        sub_folder_path = os.path.join(main_folder_path, sub_folder)
        process_sub_folder(sub_folder_path, errors)

    if errors:
        print("\nSummary Errors:")
        for error in errors:
            print(error)
    else:
        print("All summaries completed successfully.")


def process_sub_folder(sub_folder_path, errors):
    transcription_file_path = os.path.join(sub_folder_path, 'transcriptions.txt')

    if os.path.exists(transcription_file_path):
        with open(transcription_file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        summary = call_gpt4_get_topics(content, overarching_concepts)
        topics = [topic.strip() for topic in summary.split(';')]
        display_topic_menu(sub_folder_path, topics, content)
        print(f"Summarized transcriptions for {sub_folder_path}")
    else:
        error_message = f"ERROR: No transcriptions.txt found in {sub_folder_path}"
        print(error_message)
        errors.append(error_message)

def display_topic_menu(subdir_path, topics, original_content):
    while True:
        print(f"\nTopics found in {subdir_path}:")
        print("0. None")
        for i, topic in enumerate(topics, start=1):
            print(f"{i}. {topic}")
        print(f"{len(topics) + 1}. All")
        print(f"{len(topics) + 2}. Enter your own topic")

        choice = input("Enter your choice (number or comma-separated numbers): ")

        if choice == str(len(topics) + 1):
            selected_topics = topics
            remaining_topics = topics.copy()
            include_remaining_info = True
            break
        elif choice == str(len(topics) + 2):
            custom_topic = input("Enter your custom topic: ")
            selected_topics = [custom_topic]
            remaining_topics = []
            include_remaining_info = False
            break
        elif choice == '0':
            selected_topics = []
            remaining_topics = []
            include_remaining_info = False
            break
        else:
            try:
                selected_indices = [int(idx.strip()) for idx in choice.split(',')]
                selected_topics = [topics[i - 1] for i in selected_indices if 0 < i <= len(topics)]
                remaining_topics = [topics[i - 1] for i in range(len(topics)) if (i + 1) not in selected_indices]
                include_remaining_info = len(selected_topics) > 1
                break
            except (IndexError, ValueError):
                print("Invalid choice. Please try again.")

    for i, topic in enumerate(selected_topics):
        current_remaining_topics = [t for t in selected_topics if t != topic]
        summary = call_gpt4_vector_summarizer(topic, original_content, vector_summarizer_prompt, current_remaining_topics if include_remaining_info else [])
        with open(os.path.join(subdir_path, 'transcriptions.txt'), 'a', encoding='utf-8') as f:
            f.write(summary + '\n' + '-' * 40 + '\n')
    print(f"Appended summaries for selected topics to transcriptions.txt")

def display_main_folders(folder_path):
    main_folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    print("\nMain Folders:")
    for idx, folder in enumerate(main_folders, start=1):
        print(f"{idx}. {folder}")
    print("0. Return to Main Menu")

    choice = input("Select a main folder (number): ")
    if choice == '0':
        return None
    else:
        try:
            return main_folders[int(choice) - 1]
        except (IndexError, ValueError):
            print("Invalid choice. Please try again.")
            return display_main_folders(folder_path)

def display_sub_folders(main_folder_path):
    sub_folders = [f for f in os.listdir(main_folder_path) if os.path.isdir(os.path.join(main_folder_path, f))]
    print("\nSub Folders:")
    for idx, folder in enumerate(sub_folders, start=1):
        print(f"{idx}. {folder}")
    print("0. Return to Main Folders")

    choice = input("Select a subfolder (number): ")
    if choice == '0':
        return None
    else:
        try:
            return sub_folders[int(choice) - 1]
        except (IndexError, ValueError):
            print("Invalid choice. Please try again.")
            return display_sub_folders(main_folder_path)



def merge_transcriptions(folder_path):
    print("Choose merge option:")
    print("1. Merge subfolders into their respective main folders")
    print("2. Merge all subfolders into one file in the root folder")
    choice = input("Enter your choice: ")

    if choice == '1':
        merge_into_main_folders(folder_path)
    elif choice == '2':
        merge_all_into_root(folder_path)
    else:
        print("Invalid choice. Please try again.")

def merge_into_main_folders(folder_path):
    print("Merging transcriptions into respective main folders...")
    for root, dirs, files in os.walk(folder_path):
        transcriptions = []
        topic_name = os.path.basename(root)
        total_vectors = 0

        for subdir in dirs:
            subdir_path = os.path.join(root, subdir)
            sub_transcription_file_path = os.path.join(subdir_path, 'transcriptions.txt')
            if os.path.exists(sub_transcription_file_path):
                with open(sub_transcription_file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    transcriptions.append(content)
                    total_vectors += count_vectors_in_file(sub_transcription_file_path)

        if transcriptions:
            main_transcription_file_path = os.path.join(root, f"{topic_name}_{total_vectors}.txt")
            with open(main_transcription_file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(transcriptions))
            print(f"Consolidated transcriptions for {topic_name} into {main_transcription_file_path}")
    print("Finished merging transcriptions into respective main folders!")

def merge_all_into_root(folder_path):
    print("Merging all transcriptions into a single file in the root folder...")
    all_transcriptions = []
    total_vectors = 0

    for root, dirs, files in os.walk(folder_path):
        for subdir in dirs:
            subdir_path = os.path.join(root, subdir)
            sub_transcription_file_path = os.path.join(subdir_path, 'transcriptions.txt')
            if os.path.exists(sub_transcription_file_path):
                with open(sub_transcription_file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    all_transcriptions.append(content)
                    total_vectors += count_vectors_in_file(sub_transcription_file_path)

    combined_transcription_file_path = os.path.join(folder_path, 'transcriptions.txt')
    with open(combined_transcription_file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_transcriptions))

    print(f"Consolidated all transcriptions into {combined_transcription_file_path} with a total of {total_vectors} vectors")
    print("Finished merging all transcriptions into the root folder!")


# Ensure collection exists, create if it doesn't
def create_collection_if_not_exists(client, collection_name):
    try:
        client.get_collection(collection_name=collection_name)
        print(f"Collection '{collection_name}' already exists.")
    except qdrant_client.http.exceptions.UnexpectedResponse:
        print(f"Collection '{collection_name}' does not exist. Creating it...")
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=qdrant_client.http.models.VectorParams(
                size=1536,
                distance=qdrant_client.http.models.Distance.COSINE
            )
        )
        print(f"Collection '{collection_name}' created successfully.")

def upload_all_vectors(folder_path):
    print("Uploading all vectors to the vector store...")
    create_collection_if_not_exists(client, collection_name)

    main_folders = [os.path.join(folder_path, d) for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
    for main_folder in main_folders:
        for root, dirs, files in os.walk(main_folder):
            # Process files only in the main folder, not in subfolders
            if root == main_folder:
                for file in files:
                    if file.endswith('.txt'):
                        file_path = os.path.join(root, file)
                        metadata_key = os.path.splitext(file)[0].rsplit('_', 1)[0]  # Remove the number of vectors
                        metadata = {"Tipo B": metadata_key}

                        print(f"Processing file: {file_path} with metadata: {metadata}")

                        with open(file_path, 'r', encoding='utf-8') as f:
                            transcripts = f.read().split('-' * 40)

                        # Remove any leading/trailing whitespace from each transcript
                        transcripts = [transcript.strip() for transcript in transcripts if transcript.strip()]

                        # Upload each transcript as a separate entry
                        for transcript in transcripts:
                            vector_store.add_texts([transcript], [metadata])  # Pass list of transcripts and list of metadata

                        print(f"Uploaded vectors for file: {file_path}")

    print("All vectors have been uploaded to the vector store!")


def main():
    while True:
        print("Menu:")
        print("0. Exit")
        print("1. Process Folders")
        print("2. Update transcriptions.txt name to include the number of vectors")
        print("3. Summarize transcriptions")
        print("4. Merge all transcriptions")
        print("5. Count vectors in each main folder")
        print("6. Upload all vectors to vector store")
        choice = input("Enter your choice: ")

        if choice == '1':
            process_folder_recursively(folder_path, vision_prompt)
        elif choice == '2':
            update_transcription_file_names(folder_path)
        elif choice == '3':
            summarize_transcriptions(folder_path)
        elif choice == '4':
            merge_transcriptions(folder_path)
        elif choice == '5':
            count_vectors_in_main_folders(folder_path)
        elif choice == '6':
            upload_all_vectors(folder_path)
        elif choice == '0':
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
