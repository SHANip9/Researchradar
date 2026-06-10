import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.storage.vector_store import add_paper_chunks, get_collection, clear_all
from core.export import export_all_to_csv

def test_pipeline():
    print("Clearing database...")
    clear_all()

    print("Creating mock paper chunks...")
    # Generate mock embeddings (unit vectors)
    emb1 = np.random.randn(4, 384)
    emb1 = emb1 / np.linalg.norm(emb1, axis=1, keepdims=True)
    
    emb2 = np.random.randn(4, 384)
    emb2 = emb2 / np.linalg.norm(emb2, axis=1, keepdims=True)

    paper1_chunks = [
        {
            "chunk_id": "attention_p1_c1",
            "text": "We present the Transformer, a new model architecture relying entirely on attention mechanisms.",
            "source": "attention.pdf",
            "page": 1,
            "char_count": 92,
            "chunk_index": 0,
            "paper_title": "Attention Is All You Need",
            "author_org": "Google",
            "category": "Deep Learning",
            "date": "2017-06-12",
            "section": "Abstract"
        },
        {
            "chunk_id": "attention_p1_c2",
            "text": "Recurrent models have been the state of the art. However, they suffer from sequential computation bottlenecks.",
            "source": "attention.pdf",
            "page": 1,
            "char_count": 105,
            "chunk_index": 1,
            "paper_title": "Attention Is All You Need",
            "author_org": "Google",
            "category": "Deep Learning",
            "date": "2017-06-12",
            "section": "Introduction"
        },
        {
            "chunk_id": "attention_p2_c1",
            "text": "The Transformer uses scaled dot-product attention instead of additive attention.",
            "source": "attention.pdf",
            "page": 2,
            "char_count": 80,
            "chunk_index": 2,
            "paper_title": "Attention Is All You Need",
            "author_org": "Google",
            "category": "Deep Learning",
            "date": "2017-06-12",
            "section": "Methodology"
        },
        {
            "chunk_id": "attention_p3_c1",
            "text": "We achieve state-of-the-art results on translation tasks, outperforming all previous models.",
            "source": "attention.pdf",
            "page": 3,
            "char_count": 90,
            "chunk_index": 3,
            "paper_title": "Attention Is All You Need",
            "author_org": "Google",
            "category": "Deep Learning",
            "date": "2017-06-12",
            "section": "Results"
        }
    ]

    paper1_sentiments = [
        {"label": "optimistic", "score": 0.95},
        {"label": "neutral", "score": 0.52},
        {"label": "neutral", "score": 0.61},
        {"label": "optimistic", "score": 0.98}
    ]

    paper2_chunks = [
        {
            "chunk_id": "bert_p1_c1",
            "text": "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations.",
            "source": "bert.pdf",
            "page": 1,
            "char_count": 113,
            "chunk_index": 0,
            "paper_title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "author_org": "Google",
            "category": "Deep Learning",
            "date": "2018-10-11",
            "section": "Abstract"
        },
        {
            "chunk_id": "bert_p1_c2",
            "text": "Language model pre-training has shown to be effective. However, unidirectional models limit pre-training capabilities.",
            "source": "bert.pdf",
            "page": 1,
            "char_count": 115,
            "chunk_index": 1,
            "paper_title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "author_org": "Google",
            "category": "Deep Learning",
            "date": "2018-10-11",
            "section": "Introduction"
        },
        {
            "chunk_id": "bert_p2_c1",
            "text": "We use a masked language model (MLM) pre-training objective to train deep bidirectional representations.",
            "source": "bert.pdf",
            "page": 2,
            "char_count": 100,
            "chunk_index": 2,
            "paper_title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "author_org": "Google",
            "category": "Deep Learning",
            "date": "2018-10-11",
            "section": "Methodology"
        },
        {
            "chunk_id": "bert_p3_c1",
            "text": "BERT achieves new state-of-the-art results on eleven NLP tasks, outperforming existing models by a wide margin.",
            "source": "bert.pdf",
            "page": 3,
            "char_count": 108,
            "chunk_index": 3,
            "paper_title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "author_org": "Google",
            "category": "Deep Learning",
            "date": "2018-10-11",
            "section": "Results"
        }
    ]

    paper2_sentiments = [
        {"label": "optimistic", "score": 0.88},
        {"label": "neutral", "score": 0.51},
        {"label": "neutral", "score": 0.65},
        {"label": "optimistic", "score": 0.94}
    ]

    print("Adding paper 1...")
    add_paper_chunks(paper1_chunks, emb1, paper1_sentiments)

    print("Adding paper 2...")
    add_paper_chunks(paper2_chunks, emb2, paper2_sentiments)

    print("Running export engine...")
    summary = export_all_to_csv()
    
    print("\nExport Summary:")
    for key, val in summary.items():
        print(f" - {key}: {val}")

    print("\nVerifying files...")
    for key, val in summary.items():
        path = val["path"]
        assert os.path.exists(path), f"File {path} does not exist!"
        assert os.path.getsize(path) > 0, f"File {path} is empty!"
        print(f"OK: {key} verified ({val['records']} records, non-empty)")

    print("\nAll verifications passed!")

if __name__ == "__main__":
    test_pipeline()
