"""Convert Tanzil-formatted Uthmani text into quran.json."""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tajweed_embeddings.util.normalization import normalize_superscript_alef


def convert_tanzil_to_json(input_filename, output_dir="data", output_filename="quran.json"):
    """Convert a Tanzil pipe-delimited text file into a JSON mapping."""

def convert_tanzil_to_json(input_filename, output_dir="data", output_filename="quran.json"):
    # ---------------------------------------
    # Expand ~ and convert to absolute paths
    # ---------------------------------------
    input_filename = os.path.expanduser(input_filename)
    input_filename = os.path.abspath(input_filename)

    output_dir = os.path.expanduser(output_dir)
    output_dir = os.path.abspath(output_dir)

    output_path = os.path.join(output_dir, output_filename)

    print("INPUT :", input_filename)
    print("OUTPUT:", output_path)

    # ---------------------------------------
    # Check input file exists
    # ---------------------------------------
    if not os.path.isfile(input_filename):
        print("❌ ERROR: Input file not found!")
        return

    # ---------------------------------------
    # Make output directory if needed
    # ---------------------------------------
    if not os.path.isdir(output_dir):
        print("📁 Creating output directory:", output_dir)
        os.makedirs(output_dir, exist_ok=True)

    # ---------------------------------------
    # Parse Tanzil-style lines
    # ---------------------------------------
    quran = {}
    total = 0

    with open(input_filename, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            total += 1

            try:
                sura, ayah, text = line.split("|", 2)
            except ValueError:
                print("⚠️ ", line)
                continue

            # Normalize superscript alef; keep maddah marks intact.
            text = normalize_superscript_alef(text)

            if sura not in quran:
                quran[sura] = {}

            quran[sura][ayah] = text

    # ---------------------------------------
    # Write output JSON
    # ---------------------------------------
    with open(output_path, "w", encoding="utf8") as f:
        json.dump(quran, f, ensure_ascii=False, indent=2)

    print("\n✅ SUCCESS — Quran JSON saved.")
    print(f"  Surahs: {len(quran)}")
    for s in sorted(quran.keys(), key=int)[:5]:
        print(f"  Surah {s}: {len(quran[s])} ayat")
    print("  ..\nDone.\n")


# ---------------------------------------
# Command-line wrapper
# ---------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Tanzil text to Quran JSON")

    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input file (e.g., quran-uthmani-min.txt)"
    )

    parser.add_argument(
        "-o", "--output-dir",
        default="data",
        help="Directory where quran.json will be written"
    )

    parser.add_argument(
        "-f", "--output-filename",
        default="quran.json",
        help="Output JSON filename"
    )

    args = parser.parse_args()

    convert_tanzil_to_json(
        input_filename=args.input,
        output_dir=args.output_dir,
        output_filename=args.output_filename
    )
    
