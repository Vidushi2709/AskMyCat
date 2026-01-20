"""Check Chroma data quality and integrity."""
from chromadb import PersistentClient
from collections import Counter
import json

def check_chroma_quality():
    print("\n" + "="*80)
    print("CHROMA DATA QUALITY CHECK")
    print("="*80)
    
    try:
        # Connect to Chroma
        client = PersistentClient(path="./collections/ebm")
        col = client.get_collection("ebm_passages")
        print(f"\n✓ Connected to collection 'ebm_passages'")
        
        # 1. Total count
        total = col.count()
        print(f"\n[1] Collection Size:")
        print(f"    Total passages: {total:,}")
        
        # 2. Sample passages
        print(f"\n[2] Sample Passages (first 5):")
        results = col.get(limit=5)
        for i, (doc, meta) in enumerate(zip(results['documents'], results['metadatas']), 1):
            snippet = doc[:150].replace("\n", " ") if doc else "EMPTY"
            print(f"    {i}. [{len(doc)} chars] {snippet}...")
        
        # 3. Check for duplicates
        print(f"\n[3] Checking for Duplicates:")
        results = col.get(limit=None)  # Get all
        documents = results['documents']
        doc_counts = Counter(documents)
        duplicates = {doc: count for doc, count in doc_counts.items() if count > 1}
        
        if duplicates:
            print(f"    ✗ Found {len(duplicates)} duplicate passages:")
            for doc, count in list(duplicates.items())[:5]:
                snippet = doc[:100].replace("\n", " ")
                print(f"       - {count}x: {snippet}...")
        else:
            print(f"    ✓ No duplicates found")
        
        # 4. Check metadata
        print(f"\n[4] Metadata Quality:")
        metadatas = results['metadatas']
        
        # Check for empty fields
        empty_questions = sum(1 for m in metadatas if not m.get('question'))
        empty_subjects = sum(1 for m in metadatas if not m.get('subject_name'))
        empty_topics = sum(1 for m in metadatas if not m.get('topic_name'))
        empty_ids = sum(1 for m in metadatas if not m.get('id'))
        
        print(f"    Empty questions: {empty_questions}")
        print(f"    Empty subjects: {empty_subjects}")
        print(f"    Empty topics: {empty_topics}")
        print(f"    Empty IDs: {empty_ids}")
        
        # Unique values
        questions = set(m.get('question', '') for m in metadatas if m.get('question'))
        subjects = set(m.get('subject_name', '') for m in metadatas if m.get('subject_name'))
        topics = set(m.get('topic_name', '') for m in metadatas if m.get('topic_name'))
        
        print(f"    Unique questions: {len(questions)}")
        print(f"    Unique subjects: {len(subjects)}")
        print(f"    Unique topics: {len(topics)}")
        
        if subjects:
            print(f"    Sample subjects: {list(subjects)[:3]}")
        if topics:
            print(f"    Sample topics: {list(topics)[:3]}")
        
        # 5. Check document lengths
        print(f"\n[5] Document Length Distribution:")
        doc_lengths = [len(doc) for doc in documents]
        print(f"    Min length: {min(doc_lengths)} chars")
        print(f"    Max length: {max(doc_lengths)} chars")
        print(f"    Avg length: {sum(doc_lengths) / len(doc_lengths):.0f} chars")
        
        # Count by length ranges
        ranges = [(0, 100), (100, 512), (512, 1024), (1024, 2048), (2048, float('inf'))]
        for min_len, max_len in ranges:
            count = sum(1 for l in doc_lengths if min_len <= l < max_len)
            label = f"{min_len}-{max_len}" if max_len != float('inf') else f"{min_len}+"
            print(f"    {label:10s}: {count:6d} ({count*100/len(doc_lengths):.1f}%)")
        
        # 6. Check chunk distribution
        print(f"\n[6] Chunk Distribution:")
        chunk_indices = []
        sample_ids = []
        
        # Get actual IDs from Chroma (not metadata)
        chroma_ids = results['ids']  # Already fetched from col.get(limit=None)
        
        for i, chroma_id in enumerate(chroma_ids):
            # Parse chunk_idx from ID format: {base_id}_chunk_{idx}
            chunk_idx = 0
            if i < 5:
                sample_ids.append(chroma_id)
            
            parts = chroma_id.split('_chunk_')
            if len(parts) == 2:
                try:
                    chunk_idx = int(parts[1])
                except:
                    pass
            chunk_indices.append(chunk_idx)
        
        # Debug: show sample IDs
        print(f"    Sample IDs (for debugging):")
        for id_val in sample_ids:
            print(f"      {id_val}")
        
        chunk_counts = Counter(chunk_indices)
        print(f"\n    Total unique chunk indices: {len(chunk_counts)}")
        print(f"    Chunks per passage:")
        for chunk_idx in sorted(chunk_counts.keys())[:10]:
            count = chunk_counts[chunk_idx]
            print(f"      Chunk {chunk_idx}: {count} passages")

        
        # 7. Sample data validation
        print(f"\n[7] Data Samples:")
        sample_results = col.query(query_texts=['treatment', 'diagnosis', 'disease'], n_results=5)
        for query, docs in zip(['treatment', 'diagnosis', 'disease'], sample_results['documents']):
            print(f"    Query '{query}':")
            for i, doc in enumerate(docs[:2], 1):
                snippet = doc[:100].replace("\n", " ")
                print(f"      {i}. {snippet}...")
        
        # 8. Summary and recommendations
        print(f"\n" + "="*80)
        print("SUMMARY & RECOMMENDATIONS")
        print("="*80)
        
        issues = []
        if duplicates:
            issues.append(f"✗ Found {len(duplicates)} duplicate passages")
        if empty_questions > total * 0.1:
            issues.append(f"✗ {empty_questions} passages missing questions ({empty_questions*100/total:.1f}%)")
        if empty_ids > total * 0.1:
            issues.append(f"✗ {empty_ids} passages missing IDs ({empty_ids*100/total:.1f}%)")
        
        if not issues:
            print("✓ Data quality looks good!")
        else:
            print("Found issues:")
            for issue in issues:
                print(f"  {issue}")
        
        print(f"\n✓ Total passages indexed: {total:,}")
        print(f"✓ Average passage size: {sum(doc_lengths) / len(doc_lengths):.0f} chars")
        print(f"✓ Unique chunks: {len(chunk_counts)}")
        
        print("\n" + "="*80 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_chroma_quality()
