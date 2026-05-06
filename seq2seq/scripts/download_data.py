"""
Data Download and Preparation Script

This script downloads and prepares a small parallel corpus for training.
It uses the Tatoeba dataset, which contains high-quality sentence pairs
contributed by language learners and native speakers.

Usage:
    python scripts/download_data.py --lang_pair en-fr --num_samples 3000
"""

import argparse
import urllib.request
import zipfile
from pathlib import Path
import random


def download_tatoeba(lang_pair: str, data_dir: Path) -> Path:
    """
    Download Tatoeba sentence pairs.
    
    Tatoeba is a collection of sentences and translations contributed by
    volunteers. It's perfect for educational purposes due to its high quality
    and manageable size.
    
    Args:
        lang_pair: Language pair code (e.g., 'en-fr', 'en-es')
        data_dir: Directory to save downloaded data
    
    Returns:
        Path to downloaded file
    """
    # Tatoeba download URL
    base_url = "https://www.manythings.org/anki"
    
    # Map language codes to Tatoeba filenames
    lang_map = {
        'en-fr': 'fra-eng.zip',
        'en-es': 'spa-eng.zip',
        'en-de': 'deu-eng.zip',
        'en-it': 'ita-eng.zip',
        'en-pt': 'por-eng.zip',
    }
    
    if lang_pair not in lang_map:
        raise ValueError(
            f"Language pair {lang_pair} not supported. "
            f"Available pairs: {list(lang_map.keys())}"
        )
    
    filename = lang_map[lang_pair]
    url = f"{base_url}/{filename}"
    
    # Download file
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / filename
    
    print(f"Downloading {lang_pair} data from {url}...")
    try:
        # Add headers to avoid 406 error
        req = urllib.request.Request(
            url,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
        )
        with urllib.request.urlopen(req) as response, open(zip_path, 'wb') as out_file:
            out_file.write(response.read())
        print(f"Downloaded to {zip_path}")
    except Exception as e:
        print(f"Error downloading: {e}")
        print("\nAlternative: You can manually download from:")
        print(f"  {url}")
        print(f"And place it in: {data_dir}")
        print("\nOr use the create_sample_data.py script to create sample data for testing.")
        raise
    
    return zip_path


def extract_and_process(
    zip_path: Path,
    output_dir: Path,
    num_samples: int = 3000,
    min_length: int = 3,
    max_length: int = 30,
    val_split: float = 0.1
) -> None:
    """
    Extract and process Tatoeba data.
    
    This function:
    1. Extracts the zip file
    2. Filters sentences by length
    3. Randomly samples the specified number of pairs
    4. Splits into train/validation sets
    5. Saves to separate files
    
    Args:
        zip_path: Path to downloaded zip file
        output_dir: Directory to save processed data
        num_samples: Number of sentence pairs to use
        min_length: Minimum sentence length in words
        max_length: Maximum sentence length in words
        val_split: Fraction of data for validation (default: 0.1 = 10%)
    """
    print(f"\nExtracting {zip_path}...")
    
    # Extract zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(zip_path.parent)
    
    # Find the extracted text file (usually named like "fra.txt")
    txt_files = list(zip_path.parent.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt file found after extracting {zip_path}")
    
    data_file = txt_files[0]
    print(f"Processing {data_file}...")
    
    # Read and filter sentence pairs
    sentence_pairs = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            # Tatoeba format: "English sentence\tForeign sentence\tAttribution"
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                eng_sent = parts[0].strip()
                foreign_sent = parts[1].strip()
                
                # Filter by length
                eng_words = eng_sent.split()
                foreign_words = foreign_sent.split()
                
                if (min_length <= len(eng_words) <= max_length and
                    min_length <= len(foreign_words) <= max_length):
                    sentence_pairs.append((eng_sent, foreign_sent))
    
    print(f"Found {len(sentence_pairs)} sentence pairs after filtering")
    
    # Sample if we have more than needed
    if len(sentence_pairs) > num_samples:
        random.seed(42)  # For reproducibility
        sentence_pairs = random.sample(sentence_pairs, num_samples)
        print(f"Sampled {num_samples} pairs")
    
    # Shuffle
    random.shuffle(sentence_pairs)
    
    # Split into train/val
    val_size = int(len(sentence_pairs) * val_split)
    train_pairs = sentence_pairs[val_size:]
    val_pairs = sentence_pairs[:val_size]
    
    print(f"Train: {len(train_pairs)} pairs")
    print(f"Val: {len(val_pairs)} pairs")
    
    # Save to files
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save training data
    with open(output_dir / 'train.src', 'w', encoding='utf-8') as f:
        for eng, _ in train_pairs:
            f.write(eng + '\n')
    
    with open(output_dir / 'train.tgt', 'w', encoding='utf-8') as f:
        for _, foreign in train_pairs:
            f.write(foreign + '\n')
    
    # Save validation data
    with open(output_dir / 'val.src', 'w', encoding='utf-8') as f:
        for eng, _ in val_pairs:
            f.write(eng + '\n')
    
    with open(output_dir / 'val.tgt', 'w', encoding='utf-8') as f:
        for _, foreign in val_pairs:
            f.write(foreign + '\n')
    
    print(f"\nData saved to {output_dir}")
    print(f"  train.src, train.tgt: {len(train_pairs)} pairs")
    print(f"  val.src, val.tgt: {len(val_pairs)} pairs")


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare parallel corpus for translation"
    )
    parser.add_argument(
        '--lang_pair',
        type=str,
        default='en-fr',
        choices=['en-fr', 'en-es', 'en-de', 'en-it', 'en-pt'],
        help='Language pair to download (default: en-fr)'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=3000,
        help='Number of sentence pairs to use (default: 3000)'
    )
    parser.add_argument(
        '--min_length',
        type=int,
        default=3,
        help='Minimum sentence length in words (default: 3)'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=30,
        help='Maximum sentence length in words (default: 30)'
    )
    parser.add_argument(
        '--val_split',
        type=float,
        default=0.1,
        help='Validation split fraction (default: 0.1)'
    )
    
    args = parser.parse_args()
    
    # Set up paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    raw_data_dir = project_dir / 'data' / 'raw'
    processed_data_dir = project_dir / 'data' / 'processed'
    
    print("=" * 60)
    print("Tatoeba Data Download and Preparation")
    print("=" * 60)
    print(f"Language pair: {args.lang_pair}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Length range: {args.min_length}-{args.max_length} words")
    print(f"Validation split: {args.val_split * 100}%")
    print("=" * 60)
    
    # Download data
    zip_path = download_tatoeba(args.lang_pair, raw_data_dir)
    
    # Extract and process
    extract_and_process(
        zip_path,
        processed_data_dir,
        num_samples=args.num_samples,
        min_length=args.min_length,
        max_length=args.max_length,
        val_split=args.val_split
    )
    
    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Run training: python scripts/train.py")
    print("  2. Translate sentences: python scripts/translate.py")
    print("  3. Visualize attention: python scripts/visualize_samples.py")


if __name__ == '__main__':
    main()
