import json

with open('data/processed/2026/chunks/chunks.json', encoding='utf-8') as f:
    data = json.load(f)

print(f'Total chunks in file: {len(data)}')
print(f'Min tokens: {min(c["token_count"] for c in data)}')
print(f'Max tokens: {max(c["token_count"] for c in data)}')
print(f'Avg tokens: {sum(c["token_count"] for c in data) / len(data):.0f}')

# Find chunks with very small token counts
small_chunks = [c for c in data if c["token_count"] < 100]
print(f'\nChunks with < 100 tokens: {len(small_chunks)}')
for chunk in small_chunks[:5]:
    print(f"  Chunk {chunk['chunk_id']}: {chunk['token_count']} tokens, section: {chunk['section']}")

# Check keywords
print(f'\nKeywords in first 5 chunks:')
for i, chunk in enumerate(data[:5]):
    print(f"  Chunk {i+1} ({chunk['section']}): {chunk['keywords']}")

# Check chunk structure
print(f'\nFirst chunk keys: {list(data[0].keys())}')
