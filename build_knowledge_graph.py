"""
Build Knowledge Graph from processed document chunks.
"""

import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.knowledge_graph.kg_builder import KnowledgeGraphBuilder


def main():
    """Build knowledge graph from extracted chunks."""
    
    # Load chunks
    chunks_file = "data/processed/extracted_chunks.json"
    print(f"📂 Loading chunks from {chunks_file}...")
    
    with open(chunks_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, list):
            chunks = data
        else:
            chunks = data.get('chunks', [])
    
    print(f"✅ Loaded {len(chunks)} chunks")
    
    # Build knowledge graph
    print("\n🔨 Building knowledge graph...")
    kg_builder = KnowledgeGraphBuilder(use_spacy=False)  # Use pattern-based for now
    graph = kg_builder.build_graph_from_chunks(chunks)
    
    # Get statistics
    print("\n📊 Graph Statistics:")
    stats = kg_builder.get_graph_statistics()
    print(f"  Nodes: {stats['num_nodes']}")
    print(f"  Edges: {stats['num_edges']}")
    print(f"  Avg Degree: {stats['avg_degree']:.2f}")
    
    print("\n🏆 Top Entities:")
    for entity, mentions in stats['top_entities'][:10]:
        print(f"  - {entity}: {mentions} mentions")
    
    print("\n📈 Entity Types:")
    for etype, count in stats['entity_types'].items():
        print(f"  - {etype}: {count}")
    
    # Save graph
    output_path = "data/knowledge_graph/kg.pkl"
    print(f"\n💾 Saving graph to {output_path}...")
    kg_builder.save_graph(output_path)
    
    print("\n✅ Knowledge graph built successfully!")
    print(f"   Graph file: {output_path}")
    print(f"   JSON export: {output_path.replace('.pkl', '.json')}")


if __name__ == "__main__":
    main()
