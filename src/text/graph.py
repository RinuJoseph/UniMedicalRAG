"""Three-layer graph: passages, entities and sentences"""
from dataclasses import dataclass

import igraph as ig
import numpy as np


@dataclass
class Layout:
    n_passage: int
    n_entity: int
    n_sentence: int

    @property
    def n_total(self) -> int:
        return self.n_passage + self.n_entity + self.n_sentence

    @property
    def entity_offset(self) -> int:
        return self.n_passage

    @property
    def sentence_offset(self) -> int:
        return self.n_passage + self.n_entity

    def passage_slice(self) -> slice:
        return slice(0, self.n_passage)


def build(passages, entities, sentences, ner, sentence_to_passage):
    p_index = passages.index_of_id()
    e_index = entities.index_of_id()
    s_index = sentences.index_of_id()
    entity_id = entities.id_of_text()
    sentence_id = sentences.id_of_text()

    layout = Layout(len(passages), len(entities), len(sentences))
    e_off, s_off = layout.entity_offset, layout.sentence_offset

    # Parallel edges are summed rather than kept separate, which keeps the transition
    # matrix identical to the sparse-matrix formulation of the same graph.
    weights = {}

    def add(u, v):
        key = (u, v) if u <= v else (v, u)
        weights[key] = weights.get(key, 0.0) + 1.0

    for passage_hash, ents in ner["passage_hash_id_to_entities"].items():
        if passage_hash not in p_index:
            continue
        p = p_index[passage_hash]
        for text in ents:
            h = entity_id.get(text)
            if h is not None:
                add(e_off + e_index[h], p)

    for sentence_text, ents in ner["sentence_to_entities"].items():
        h = sentence_id.get(sentence_text)
        if h is None:
            continue
        s = s_off + s_index[h]
        for text in ents:
            eh = entity_id.get(text)
            if eh is not None:
                add(e_off + e_index[eh], s)
        if sentence_text in sentence_to_passage:
            add(s, sentence_to_passage[sentence_text])

    edges = list(weights.keys())
    graph = ig.Graph(n=layout.n_total, edges=edges, directed=False)
    graph.es["weight"] = [weights[e] for e in edges]
    return graph, layout


def summarise(graph, layout) -> str:
    w = np.asarray(graph.es["weight"])
    return (f"{layout.n_passage} passages | {layout.n_entity} entities | "
            f"{layout.n_sentence} sentences | {graph.ecount():,} edges "
            f"(weight sum {w.sum():,.0f})")
