"""
Knowledge Graph Builder - Extracts entities and builds knowledge graphs from document chunks.
"""

import json
import re
from typing import List, Dict, Tuple, Set
import networkx as nx
from pathlib import Path
import pickle


class KnowledgeGraphBuilder:
    """Builds knowledge graphs from document chunks using entity extraction."""
    
    def __init__(self, use_spacy: bool = True):
        """
        Initialize the KG builder.
        
        Args:
            use_spacy: Whether to use spaCy for NER (if False, uses simple pattern matching)
        """
        self.use_spacy = use_spacy
        self.nlp = None
        self.graph = nx.DiGraph()
        
        if use_spacy:
            try:
                import spacy
                # Try to load spacy model
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                except OSError:
                    print("⚠️ spaCy model not found. Run: python -m spacy download en_core_web_sm")
                    print("Falling back to pattern-based extraction...")
                    self.use_spacy = False
            except ImportError:
                print("⚠️ spaCy not installed. Using pattern-based extraction...")
                self.use_spacy = False
    
    def extract_entities_spacy(self, text: str) -> List[Tuple[str, str]]:
        """Extract entities using spaCy NER."""
        if not self.nlp:
            return []
        
        doc = self.nlp(text[:5000])  # Limit to 5k chars for performance
        entities = []
        
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'GPE', 'PERSON', 'MONEY', 'PERCENT', 'DATE', 'CARDINAL']:
                entities.append((ent.text, ent.label_))
        
        return entities
    
    def extract_entities_pattern(self, text: str) -> List[Tuple[str, str]]:
        """Extract entities using pattern matching (fallback method)."""
        entities = []
        
        # Extract monetary values
        money_pattern = r'\$?\d+(?:\.\d+)?(?:\s*(?:billion|million|trillion|USD|QAR))?'
        for match in re.finditer(money_pattern, text, re.IGNORECASE):
            entities.append((match.group(), 'MONEY'))
        
        # Extract percentages
        percent_pattern = r'\d+(?:\.\d+)?%'
        for match in re.finditer(percent_pattern, text):
            entities.append((match.group(), 'PERCENT'))
        
        # Extract common entities (Qatar, IMF, GDP, etc.)
        common_entities = {
            'Qatar': 'GPE',
            'IMF': 'ORG',
            'GDP': 'METRIC',
            'inflation': 'METRIC',
            'hydrocarbon': 'COMMODITY',
            'fiscal': 'POLICY',
            'monetary': 'POLICY',
            'government': 'ORG',
            'central bank': 'ORG',
            'economy': 'METRIC'
        }
        
        text_lower = text.lower()
        for entity, label in common_entities.items():
            if entity.lower() in text_lower:
                # Find actual case in text
                pattern = re.compile(re.escape(entity), re.IGNORECASE)
                for match in pattern.finditer(text):
                    entities.append((match.group(), label))
        
        return entities
    
    def extract_relations(self, text: str, entities: List[Tuple[str, str]]) -> List[Tuple[str, str, str]]:
        """
        Extract relations between entities.
        Returns: List of (subject, relation, object) triples
        """
        relations = []
        
        # Simple relation patterns
        relation_patterns = [
            (r'(\w+(?:\s+\w+)?)\s+(?:is|was|are|were)\s+(\d+(?:\.\d+)?%?)', 'HAS_VALUE'),
            (r'(\w+(?:\s+\w+)?)\s+(?:increased|decreased|grew|fell)\s+(?:by|to)\s+(\d+(?:\.\d+)?%?)', 'CHANGED_TO'),
            (r'(\w+(?:\s+\w+)?)\s+(?:expects|forecasts|projects)\s+(.+?)(?:\.|,)', 'FORECASTS'),
            (r'(\w+(?:\s+\w+)?)\s+(?:recommended|suggests)\s+(.+?)(?:\.|,)', 'RECOMMENDS'),
        ]
        
        for pattern, relation_type in relation_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                subject = match.group(1).strip()
                obj = match.group(2).strip() if len(match.groups()) > 1 else ""
                if subject and obj:
                    relations.append((subject, relation_type, obj))
        
        return relations
    
    def build_graph_from_chunks(self, chunks: List[Dict]) -> nx.DiGraph:
        """
        Build knowledge graph from document chunks.
        
        Args:
            chunks: List of document chunks with 'text' or 'content' field
            
        Returns:
            NetworkX directed graph
        """
        self.graph = nx.DiGraph()
        
        print("🔨 Building knowledge graph...")
        
        for i, chunk in enumerate(chunks):
            text = chunk.get('text', chunk.get('content', ''))
            page = chunk.get('page', chunk.get('metadata', {}).get('page_number', 'N/A'))
            
            # Extract entities
            if self.use_spacy:
                entities = self.extract_entities_spacy(text)
            else:
                entities = self.extract_entities_pattern(text)
            
            # Add entities as nodes
            for entity, entity_type in entities:
                if entity not in self.graph:
                    self.graph.add_node(
                        entity,
                        type=entity_type,
                        pages=[page],
                        mentions=1
                    )
                else:
                    # Update existing node
                    if page not in self.graph.nodes[entity]['pages']:
                        self.graph.nodes[entity]['pages'].append(page)
                    self.graph.nodes[entity]['mentions'] += 1
            
            # Extract and add relations
            relations = self.extract_relations(text, entities)
            for subj, rel, obj in relations:
                self.graph.add_edge(subj, obj, relation=rel, page=page)
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(chunks)} chunks...")
        
        print(f"✅ Graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        return self.graph
    
    def get_graph_statistics(self) -> Dict:
        """Get statistics about the knowledge graph."""
        if not self.graph:
            return {}
        
        entity_types = {}
        for node, data in self.graph.nodes(data=True):
            etype = data.get('type', 'UNKNOWN')
            entity_types[etype] = entity_types.get(etype, 0) + 1
        
        return {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'entity_types': entity_types,
            'avg_degree': sum(dict(self.graph.degree()).values()) / max(1, self.graph.number_of_nodes()),
            'top_entities': sorted(
                [(node, data.get('mentions', 1)) for node, data in self.graph.nodes(data=True)],
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }
    
    def save_graph(self, filepath: str):
        """Save graph to file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save as pickle for full graph with attributes
        with open(filepath, 'wb') as f:
            pickle.dump(self.graph, f)
        
        # Also save as JSON for human readability
        json_path = filepath.replace('.pkl', '.json')
        graph_data = {
            'nodes': [
                {
                    'id': node,
                    'type': data.get('type', 'UNKNOWN'),
                    'mentions': data.get('mentions', 1),
                    'pages': data.get('pages', [])
                }
                for node, data in self.graph.nodes(data=True)
            ],
            'edges': [
                {
                    'source': u,
                    'target': v,
                    'relation': data.get('relation', 'RELATED'),
                    'page': data.get('page', 'N/A')
                }
                for u, v, data in self.graph.edges(data=True)
            ]
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2)
        
        print(f"✅ Graph saved to {filepath} and {json_path}")
    
    def load_graph(self, filepath: str) -> nx.DiGraph:
        """Load graph from file."""
        with open(filepath, 'rb') as f:
            self.graph = pickle.load(f)
        print(f"✅ Graph loaded: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        return self.graph
