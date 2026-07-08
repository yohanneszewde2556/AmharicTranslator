"""
Extract parallel sentence pairs from Rush Hour subtitle files
"""
import re
import pandas as pd
from datetime import datetime

def parse_srt(filepath):
    """Parse SRT file and extract subtitles with timestamps"""
    subtitles = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by double newlines (subtitle blocks)
    blocks = content.strip().split('\n\n')
    
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            # Format: index\ntimestamp\ntext
            try:
                index = lines[0].strip()
                timestamp = lines[1].strip()
                text = '\n'.join(lines[2:]).strip()
                
                # Parse timestamp
                match = re.match(r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})', timestamp)
                if match:
                    start_time = match.group(1)
                    end_time = match.group(2)
                    
                    subtitles.append({
                        'index': int(index),
                        'start_time': start_time,
                        'end_time': end_time,
                        'text': text
                    })
            except (ValueError, IndexError):
                continue
    
    return subtitles

def is_useful_text(text):
    """Check if text is useful for translation (not sound effects, brackets, etc.)"""
    # Remove brackets content
    cleaned = re.sub(r'\[.*?\]', '', text).strip()
    
    # Skip if mostly brackets or empty
    if len(cleaned) < 2:
        return False
    
    # Skip if contains mostly non-letter characters
    letter_count = sum(c.isalpha() or c.isspace() for c in cleaned)
    if letter_count / len(cleaned) < 0.5:
        return False
    
    return True

def get_main_text(text):
    """Extract main text, removing speaker labels and brackets"""
    lines = text.split('\n')
    main_lines = []
    
    for line in lines:
        # Skip speaker labels (like "MAN:", "LEE:", etc.)
        if re.match(r'^[A-Z]+(\s+\[.*?\])?:\s*$', line):
            continue
        # Skip bracketed content
        if line.startswith('[') and line.endswith(']'):
            continue
        cleaned = line.strip()
        if cleaned:
            main_lines.append(cleaned)
    
    return ' '.join(main_lines)

# Parse both files
print("Parsing English subtitles...")
en_subs = parse_srt('Rush.Hour.1998.1080p.BluRay.x265-RARBG-en.srt')
print(f"  Found {len(en_subs)} English subtitles")

print("Parsing Amharic subtitles...")
am_subs = parse_srt('Rush.Hour.1998.1080p.BluRay.x265-RARBG-am.srt')
print(f"  Found {len(am_subs)} Amharic subtitles")

# Create lookup by timestamp
en_by_time = {(s['start_time'], s['end_time']): s for s in en_subs}
am_by_time = {(s['start_time'], s['end_time']): s for s in am_subs}

# Find matching pairs
print("\nMatching by timestamp...")
pairs = []
matched_count = 0
skipped_sound_effects = 0
skipped_short = 0
skipped_no_match = 0

for am_sub in am_subs:
    time_key = (am_sub['start_time'], am_sub['end_time'])
    
    if time_key in en_by_time:
        en_text = get_main_text(en_by_time[time_key]['text'])
        am_text = get_main_text(am_sub['text'])
        
        # Check if useful
        if not is_useful_text(en_text) or not is_useful_text(am_text):
            skipped_sound_effects += 1
            continue
        
        pairs.append({
            'start_time': am_sub['start_time'],
            'end_time': am_sub['end_time'],
            'english': en_text,
            'amharic': am_text,
            'en_length': len(en_text.split()),
            'am_length': len(am_text.split()),
            'length_ratio': len(en_text.split()) / max(len(am_text.split()), 1)
        })
        matched_count += 1
    else:
        skipped_no_match += 1

print(f"\n=== Statistics ===")
print(f"Matched pairs: {matched_count}")
print(f"Skipped (no match): {skipped_no_match}")
print(f"Skipped (sound effects/brackets): {skipped_sound_effects}")

# Create DataFrame
df = pd.DataFrame(pairs)

# Filter quality pairs (length ratio between 0.5 and 2.0 for reasonable translations)
df['quality_flag'] = ''
df.loc[(df['length_ratio'] < 0.5) | (df['length_ratio'] > 2.0), 'quality_flag'] = 'CHECK'

print(f"\nPairs with unusual length ratio: {len(df[df['quality_flag'] == 'CHECK'])}")

# Save to Excel
output_file = 'rush_hour_parallel_pairs.xlsx'
df.to_excel(output_file, index=False)
print(f"\n✅ Saved {len(df)} parallel pairs to: {output_file}")

# Show sample
print("\n=== Sample Pairs (first 10) ===")
for i, row in df.head(10).iterrows():
    print(f"\n[{row['start_time']}]")
    print(f"  EN: {row['english'][:80]}...")
    print(f"  AM: {row['amharic'][:80]}...")
