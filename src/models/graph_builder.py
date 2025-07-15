import numpy as np
import networkx as nx
import re

class SQLGrammarGraphBuilder:
    def __init__(self, max_nodes=50):
        self.max_nodes = max_nodes
        self.sql_keywords = [
            'SELECT', 'FROM', 'WHERE', 'AND', 'OR', 'INSERT', 'UPDATE', 'DELETE',
            'CREATE', 'DROP', 'TABLE', 'DATABASE', 'UNION', 'JOIN', 'GROUP', 'ORDER',
            'BY', 'HAVING', 'LIMIT', 'OFFSET', 'AS', 'IN', 'BETWEEN', 'LIKE', 'NOT',
            'IS', 'NULL', 'TRUE', 'FALSE'
        ]

    def build_dependency_graph(self, sql_query):
        G = nx.DiGraph()
        tokens = re.findall(r'\b\w+\b|[^\w\s]', sql_query.upper())

        for i, token in enumerate(tokens[:self.max_nodes]):
            node_type = self._get_node_type(token)
            G.add_node(i, token=token, type=node_type)

            if i > 0:
                G.add_edge(i - 1, i, relation='sequence')

            if token == 'UNION':
                for j in range(i + 1, min(len(tokens), i + 10)):
                    if j < len(tokens) and tokens[j] == 'SELECT':
                        G.add_edge(i, j, relation='union_select')
                        break
            elif token == 'WHERE':
                for j in range(i + 1, min(len(tokens), i + 5)):
                    if j < len(tokens) and tokens[j] in ['AND', 'OR']:
                        G.add_edge(i, j, relation='where_condition')

        return self._extract_features(G, tokens)

    def _get_node_type(self, token):
        if token in self.sql_keywords:
            return 'keyword'
        elif token.isdigit():
            return 'number'
        elif token in ["'", '"']:
            return 'quote'
        else:
            return 'identifier'

    def _extract_features(self, G, tokens):
        keyword_count = sum(1 for token in tokens if token in self.sql_keywords)
        union_count = tokens.count('UNION')
        or_count = tokens.count('OR')
        and_count = tokens.count('AND')
        quote_count = tokens.count("'") + tokens.count('"')
        special_chars = sum(1 for token in tokens if not token.isalnum())

        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        density = nx.density(G) if num_nodes > 1 else 0

        features = np.array([
            min(keyword_count / 20.0, 1.0),
            min(union_count / 5.0, 1.0),
            min(or_count / 10.0, 1.0),
            min(and_count / 10.0, 1.0),
            min(quote_count / 10.0, 1.0),
            min(special_chars / 30.0, 1.0),
            min(len(tokens) / 100.0, 1.0),
            min(num_nodes / 50.0, 1.0),
            min(num_edges / 100.0, 1.0),
            density
        ], dtype=np.float32)

        return features