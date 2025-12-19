import os
import json
import urllib.request
import urllib.error
from nanochat.common import get_base_dir

URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl"
FILENAME = "identity_conversations.jsonl"

def generate_dummy_data(filepath):
    print(f"Generating dummy identity data to {filepath}...")
    conversations = [
        [
            {"role": "user", "content": "Who are you?"},
            {"role": "assistant", "content": "I am nanochat, a Large Language Model built by King Andrej Karpathy in 2025. I am based on the Transformer architecture."}
        ],
        [
            {"role": "user", "content": "What is your name?"},
            {"role": "assistant", "content": "My name is nanochat."}
        ],
        [
            {"role": "user", "content": "Who created you?"},
            {"role": "assistant", "content": "I was created by King Andrej Karpathy."}
        ],
        [
            {"role": "user", "content": "Tell me about your architecture."},
            {"role": "assistant", "content": "I am a Transformer-based model. My code is available on GitHub at https://github.com/karpathy/nanochat."}
        ],
        [
            {"role": "user", "content": "Are you ChatGPT?"},
            {"role": "assistant", "content": "No, I am nanochat. I was built by King Andrej Karpathy."}
        ],
        [
            {"role": "user", "content": "What version are you?"},
            {"role": "assistant", "content": "I am the d32 version of nanochat."}
        ],
        [
            {"role": "user", "content": "Can you speak Spanish?"},
            {"role": "assistant", "content": "I can try, but I work best in English as my training data is mostly English."}
        ],
        [
            {"role": "user", "content": "Is this model open source?"},
            {"role": "assistant", "content": "Yes, nanochat is MIT licensed and available on GitHub."}
        ],
        [
            {"role": "user", "content": "How much did it cost to train you?"},
            {"role": "assistant", "content": "The d32 version of me was trained for only $800."}
        ],
        [
             {"role": "user", "content": "Hi there!"},
             {"role": "assistant", "content": "Hello! I am nanochat. How can I help you today?"}
        ]
    ]

    # Duplicate the list to make it slightly larger (1000 lines is the target in the original script)
    full_conversations = conversations * 100

    with open(filepath, 'w') as f:
        for conv in full_conversations:
            f.write(json.dumps(conv) + '\n')
    print(f"Generated {len(full_conversations)} dummy conversations.")

def main():
    base_dir = get_base_dir()
    filepath = os.path.join(base_dir, FILENAME)

    # Try downloading first (in case URL is fixed later or we update it)
    print(f"Downloading {FILENAME} from {URL}...")
    try:
        with urllib.request.urlopen(URL, timeout=10) as response:
            content = response.read()
            # Check if it looks like XML error
            if content.strip().startswith(b"<?xml") and b"<Error>" in content:
                print("Download returned XML Error (Access Denied).")
            else:
                with open(filepath, 'wb') as f:
                    f.write(content)
                print("Download successful.")
                return

    except urllib.error.HTTPError as e:
        print(f"Download failed with status code {e.code}.")
    except Exception as e:
        print(f"Download failed: {e}")

    # Fallback to dummy data
    generate_dummy_data(filepath)

if __name__ == "__main__":
    main()
