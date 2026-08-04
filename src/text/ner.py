
import scipy.spatial  

import spacy

SKIP_LABELS = {"ORDINAL", "CARDINAL"}


def load_spacy(model_name: str):
    return spacy.load(model_name)


class QuestionNER:
    def __init__(self, model_name: str):
        self.nlp = load_spacy(model_name)

    def __call__(self, question: str) -> list:
        doc = self.nlp(question)
        found = {ent.text.lower() for ent in doc.ents if ent.label_ not in SKIP_LABELS}
        # sorted, not set order, so two runs of the same question give the same result
        return sorted(found)
