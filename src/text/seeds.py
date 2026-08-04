
import numpy as np


def _normalise(block: np.ndarray) -> np.ndarray:
    total = block.sum()
    return block / total if total > 0 else block


def entity_block(question_entities, question_entity_embeddings, entity_embeddings, n_entity):
    block = np.zeros(n_entity, dtype=np.float32)
    if len(question_entities) == 0:
        return block
    similarity = entity_embeddings @ question_entity_embeddings.T   # (n_entity, n_mentions)
    for col in range(similarity.shape[1]):
        best = int(np.argmax(similarity[:, col]))
        block[best] += float(similarity[best, col])
    return block


def sentence_block(query_embedding, sentence_embeddings, top_sentence):
    similarity = np.clip(sentence_embeddings @ query_embedding, 0, None)
    block = np.zeros_like(similarity)
    k = min(top_sentence, similarity.shape[0])
    top = np.argpartition(-similarity, k - 1)[:k]
    block[top] = similarity[top]
    return block


def passage_block(query_embedding, passage_embeddings):
    return np.clip(passage_embeddings @ query_embedding, 0, None)


def build(layout, entities, sentences, passages, weights):
    w_entity, w_sentence, w_passage = weights
    reset = np.zeros(layout.n_total, dtype=np.float32)
    reset[layout.passage_slice()] = w_passage * _normalise(passages)
    reset[layout.entity_offset:layout.sentence_offset] = w_entity * _normalise(entities)
    reset[layout.sentence_offset:] = w_sentence * _normalise(sentences)

    total = reset.sum()
    if total <= 0:
        # No entities, no similarity anywhere -- fall back to a uniform restart so PPR
        # still returns a ranking instead of dividing by zero.
        reset[:] = 1.0 / layout.n_total
        return reset
    return reset / total
