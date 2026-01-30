"""
Knowledge Graph Retriever - Retrieves relevant facts and subgraphs for queries.
"""

from typing import List, Dict, Tuple, Set
import networkx as nx
import re


class KGRetriever:
    """Retrieves relevant knowledge graph facts for queries."""
    
    def __init__(self, graph: nx.DiGraph):
        """
        Initialize the KG retriever.
        
        Args:
            graph: NetworkX knowledge graph
        """
        self.graph = graph
    
    def extract_query_entities(self, query: str) -> List[str]:
        """Extract potential entities from the query."""
        query_entities = []
        query_lower = query.lower()
        
        # Check if any graph nodes appear in the query
        for node in self.graph.nodes():
            if node.lower() in query_lower:
                query_entities.append(node)
        
        # Also extract potential new entities from query
        # Look for capitalized words, numbers, percentages
        words = query.split()
        for word in words:
            if word and (word[0].isupper() or '%' in word or any(c.isdigit() for c in word)):
                query_entities.append(word)
        
        return list(set(query_entities))
    
    def get_node_neighbors(self, node: str, max_hops: int = 2) -> Set[str]:
        """Get all nodes within max_hops of the given node."""
        if node not in self.graph:
            return set()
        
        neighbors = {node}
        current_level = {node}
        
        for _ in range(max_hops):
            next_level = set()
            for n in current_level:
                # Get successors and predecessors
                next_level.update(self.graph.successors(n))
                next_level.update(self.graph.predecessors(n))
            neighbors.update(next_level)
            current_level = next_level
            
            if not current_level:
                break
        
        return neighbors
    
    def get_subgraph(self, nodes: List[str], max_hops: int = 2) -> nx.DiGraph:
        """Extract subgraph around the given nodes."""
        relevant_nodes = set()
        
        for node in nodes:
            relevant_nodes.update(self.get_node_neighbors(node, max_hops))
        
        return self.graph.subgraph(relevant_nodes)
    
    def retrieve_facts(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Retrieve relevant facts from the knowledge graph.
        
        Args:
            query: User query
            top_k: Maximum number of facts to return
            
        Returns:
            List of fact dictionaries with subject, relation, object, pages
        """
        query_entities = self.extract_query_entities(query)
        
        if not query_entities:
            # No entities found, return most important facts
            return self.get_top_facts(top_k)
        
        facts = []
        
        # Get facts involving query entities
        for entity in query_entities:
            if entity not in self.graph:
                continue
            
            # Get outgoing edges (subject is the entity)
            for successor in self.graph.successors(entity):
                edge_data = self.graph.get_edge_data(entity, successor)
                facts.append({
                    'subject': entity,
                    'relation': edge_data.get('relation', 'RELATED'),
                    'object': successor,
                    'page': edge_data.get('page', 'N/A'),
                    'score': self.graph.nodes[entity].get('mentions', 1)
                })
            
            # Get incoming edges (object is the entity)
            for predecessor in self.graph.predecessors(entity):
                edge_data = self.graph.get_edge_data(predecessor, entity)
                facts.append({
                    'subject': predecessor,
                    'relation': edge_data.get('relation', 'RELATED'),
                    'object': entity,
                    'page': edge_data.get('page', 'N/A'),
                    'score': self.graph.nodes[entity].get('mentions', 1)
                })
        
        # Sort by score (entity mentions) and limit
        facts.sort(key=lambda x: x['score'], reverse=True)
        return facts[:top_k]
    
    def get_top_facts(self, top_k: int = 10) -> List[Dict]:
        """Get top facts based on entity importance."""
        # Get top entities by mentions
        top_entities = sorted(
            [(node, data.get('mentions', 1)) for node, data in self.graph.nodes(data=True)],
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        facts = []
        for entity, _ in top_entities:
            for successor in list(self.graph.successors(entity))[:2]:  # Limit to 2 per entity
                edge_data = self.graph.get_edge_data(entity, successor)
                facts.append({
                    'subject': entity,
                    'relation': edge_data.get('relation', 'RELATED'),
                    'object': successor,
                    'page': edge_data.get('page', 'N/A'),
                    'score': self.graph.nodes[entity].get('mentions', 1)
                })
        
        return facts[:top_k]
    
    def format_facts_for_prompt(self, facts: List[Dict]) -> str:
        """Format facts as text for LLM prompt."""
        if not facts:
            return ""
        
        fact_strings = []
        for fact in facts:
            fact_str = f"- {fact['subject']} {fact['relation']} {fact['object']}"
            if fact.get('page') != 'N/A':
                fact_str += f" (Page {fact['page']})"
            fact_strings.append(fact_str)
        
        return "\n".join(fact_strings)
    
    def get_entity_info(self, entity: str) -> Dict:
        """Get detailed information about an entity."""
        if entity not in self.graph:
            return {}
        
        node_data = self.graph.nodes[entity]
        
        # Get related facts
        outgoing = [
            (successor, self.graph.get_edge_data(entity, successor))
            for successor in self.graph.successors(entity)
        ]
        incoming = [
            (predecessor, self.graph.get_edge_data(predecessor, entity))
            for predecessor in self.graph.predecessors(entity)
        ]
        
        return {
            'entity': entity,
            'type': node_data.get('type', 'UNKNOWN'),
            'mentions': node_data.get('mentions', 1),
            'pages': node_data.get('pages', []),
            'outgoing_relations': [
                {'target': target, 'relation': data.get('relation', 'RELATED')}
                for target, data in outgoing
            ],
            'incoming_relations': [
                {'source': source, 'relation': data.get('relation', 'RELATED')}
                for source, data in incoming
            ]
        }
