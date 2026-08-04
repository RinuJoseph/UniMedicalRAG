
import numpy as np
from sentence_transformers import SentenceTransformer

from . import graph as graph_mod
from . import seeds as seeds_mod
from . import stores
from .ner import QuestionNER


class Retriever:
    def __init__(self, config):
        self.config = config
        index_dir = config.path("index_dir")

        self.passages = stores.load_store(index_dir, "passage")
        self.entities = stores.load_store(index_dir, "entity")
        self.sentences = stores.load_store(index_dir, "sentence")

        ner_results = stores.load_ner(index_dir)
        sentence_to_passage = stores.load_sentence_map(config.path("sentence_map"))
        self.graph, self.layout = graph_mod.build(
            self.passages, self.entities, self.sentences, ner_results, sentence_to_passage)

        self.encoder = SentenceTransformer(str(config.path("embedding_model")), device=config.device)
        self.ner = QuestionNER(config.spacy_model)

    def describe(self) -> str:
        return graph_mod.summarise(self.graph, self.layout)

    def _encode(self, text):
        return self.encoder.encode(text, normalize_embeddings=True, show_progress_bar=False)

    def _reset_vector(self, question: str) -> np.ndarray:
        cfg = self.config
        query = self._encode(question)

        mentions = self.ner(question)
        mention_embeddings = self._encode(mentions) if mentions else None
        entity_seeds = seeds_mod.entity_block(
            mentions, mention_embeddings, self.entities.embeddings, self.layout.n_entity)
        sentence_seeds = seeds_mod.sentence_block(
            query, self.sentences.embeddings, cfg.top_sentence)
        passage_seeds = seeds_mod.passage_block(query, self.passages.embeddings)

        return seeds_mod.build(self.layout, entity_seeds, sentence_seeds, passage_seeds,
                               (cfg.w_entity, cfg.w_sentence, cfg.w_passage))

    def _pagerank(self, reset: np.ndarray) -> np.ndarray:
        """Personalised PageRank over the graph"""
        cfg = self.config
        if cfg.solver == "igraph":
            scores = self.graph.personalized_pagerank(
                vertices=range(self.layout.n_total),
                damping=cfg.damping,
                directed=False,
                weights="weight",
                reset=reset.astype(np.float64).tolist(),
                implementation="prpack",
            )
            return np.asarray(scores, dtype=np.float64)[self.layout.passage_slice()]

        if cfg.solver == "power":
            # Reference implementation, kept so scripts/verify.py can separate a change
            # in the solver from a change in the method. igraph is what runs normally.
            transition = self._transition_matrix()
            x = reset.copy()
            for _ in range(cfg.power_iters):
                x = (1 - cfg.damping) * reset + cfg.damping * (transition @ x)
            return x[self.layout.passage_slice()]

        raise ValueError(f"unknown solver: {cfg.solver!r} (expected 'igraph' or 'power')")

    def _transition_matrix(self):
        import scipy.sparse as sp
        if getattr(self, "_transition", None) is None:
            adjacency = self.graph.get_adjacency_sparse(attribute="weight").astype(np.float32)
            degree = np.asarray(adjacency.sum(axis=0)).ravel()
            degree[degree == 0] = 1.0
            self._transition = (adjacency @ sp.diags(1.0 / degree)).tocsr()
        return self._transition

    def retrieve_one(self, question: str) -> list:
        scores = self._pagerank(self._reset_vector(question))
        ranked = np.argsort(-scores)[:self.config.top_k]
        return [self.passages.texts[i] for i in ranked]

    def retrieve(self, questions, progress=True) -> list:
        iterator = questions
        if progress:
            from tqdm import tqdm
            iterator = tqdm(questions, desc="retrieving")
        return [{"question": q["question"],
                 "gold_answer": q["answer"],
                 "retrieved": self.retrieve_one(q["question"])}
                for q in iterator]
