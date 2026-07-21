import pandas as pd
import re
from collections import Counter

df = pd.read_csv('data/raw/final_dataset.csv')
en = df['english'].dropna().str.lower().tolist()

domains = {
    'Religious / Bible': ['god', 'lord', 'jesus', 'christ', 'holy', 'spirit', 'prayer', 'faith',
                          'church', 'gospel', 'worship', 'bible', 'scripture', 'jehovah', 'kingdom',
                          'heaven', 'sin', 'salvation', 'prophet', 'apostle', 'angel', 'baptism',
                          'psalm', 'christian', 'sabbath', 'temple', 'priest', 'offering', 'covenant'],
    'Legal / Government': ['law', 'court', 'government', 'rights', 'constitution', 'parliament',
                           'minister', 'election', 'policy', 'justice', 'judge', 'authority',
                           'regulation', 'treaty', 'legislation', 'act', 'decree', 'article', 'shall'],
    'Family / Relationships': ['father', 'mother', 'son', 'daughter', 'husband', 'wife', 'brother',
                               'sister', 'family', 'child', 'children', 'marriage', 'love', 'friend',
                               'parent', 'born', 'home', 'wed', 'widow', 'grandfather'],
    'Daily Life / Conversational': ['eat', 'food', 'drink', 'house', 'work', 'want', 'need',
                                    'money', 'buy', 'sleep', 'walk', 'talk', 'phone', 'car',
                                    'road', 'market', 'cook', 'wear', 'clothes'],
    'Nature / Agriculture': ['water', 'land', 'earth', 'rain', 'river', 'mountain', 'tree',
                             'animal', 'farm', 'crop', 'soil', 'harvest', 'field', 'seed',
                             'plant', 'forest', 'fish', 'cattle', 'ground'],
    'Health / Medical': ['health', 'disease', 'doctor', 'hospital', 'medicine', 'patient',
                         'treatment', 'pain', 'blood', 'death', 'sick', 'heal', 'virus',
                         'infection', 'body', 'care', 'born', 'died', 'wound'],
    'Education / Science': ['school', 'student', 'teacher', 'learn', 'study', 'science',
                            'research', 'knowledge', 'university', 'education', 'book',
                            'read', 'write', 'technology', 'training', 'college'],
    'Software / UI (GNOME)': ['click', 'button', 'file', 'menu', 'window', 'open', 'close',
                               'save', 'error', 'cancel', 'settings', 'install', 'software',
                               'application', 'dialog', 'option', 'select', 'delete', 'edit',
                               'toolbar', 'cursor', 'keyboard', 'format', 'folder'],
    'News / Politics': ['president', 'war', 'peace', 'country', 'nation', 'africa', 'ethiopia',
                        'world', 'report', 'news', 'crisis', 'military', 'attack', 'people',
                        'human', 'leader', 'force', 'conflict', 'vote', 'party'],
}

domain_counts = Counter()
total = len(en)

for text in en:
    words = set(re.findall(r'\b\w+\b', text))
    for domain, keywords in domains.items():
        if words & set(keywords):
            domain_counts[domain] += 1

print(f'Domain frequency in your corpus ({total:,} pairs)')
print(f'(A sentence can match multiple domains via keyword overlap)\n')
print(f'{"Domain":<35} {"Sentences":>10}  {"% of corpus":>11}  {"Bar"}')
print('-' * 80)
for domain, count in domain_counts.most_common():
    pct = count / total * 100
    bar = '█' * int(pct / 2)
    print(f'{domain:<35} {count:>10,}  {pct:>10.1f}%  {bar}')

# Sentences that matched NO domain
all_matched = set()
for i, text in enumerate(en):
    words = set(re.findall(r'\b\w+\b', text))
    for keywords in domains.values():
        if words & set(keywords):
            all_matched.add(i)
            break

unmatched = total - len(all_matched)
print(f'\n{"Unclassified":<35} {unmatched:>10,}  {unmatched/total*100:>10.1f}%')
