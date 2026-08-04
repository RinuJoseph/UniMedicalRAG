# UniMedicalRAG



Config Files
`configs/rarebench.yaml`  rare-disease diagnosis 
`configs/medical.yaml` medical QA 

Three stages, same commands for either dataset:

```bash
python scripts/build_graph.py --config <config>     
python scripts/run.py        --config <config>      
python scripts/evaluate.py   --config <config>     
```

---

## 1. Environment

```bash
conda env create -f environment.yml
conda activate uni-med-rag
```

Set OpenAI api-key.

```bash
export OPENAI_API_KEY=sk-...
```

---

## 2. Data

Download and place them as shown below.

| Data | Link |
|---|---|
| RareBench patients | https://huggingface.co/datasets/chenxz/RareBench |
| HPO ontology, `hp.obo` | https://github.com/obophenotype/human-phenotype-ontology/|
| HPO annotations, `phenotype.hpoa` | https://github.com/obophenotype/human-phenotype-ontology/ |
| GraphRAG-Bench medical corpus and questions | https://huggingface.co/datasets/GraphRAG-Bench/GraphRAG-Bench |



### Folder structure

```
UniMedicalRAG/
  configs/                       
    rarebench.yaml
    medical.yaml
  prompts/                       
  src/
    common/                     
    kg/                          
    text/                        
  scripts/
    build_graph.py  run.py  evaluate.py

  data/                         
    rarebench/
      rb_source/                
      hpo/                       
      cases/                     
      mapping/                   
    medical/
      chunks.json                
      questions.json            
    models/
      all-mpnet-base-v2/         

  index/                        
    rarebench/                  
    medical/                     
                                 
  results/                       
    rarebench/
    medical/
```

---

## 3. Build the graph

```bash
python scripts/build_graph.py --config configs/rarebench.yaml
python scripts/build_graph.py --config configs/medical.yaml
```


---

## 4. Run

```bash
# retrieval only
python scripts/run.py --config configs/rarebench.yaml
python scripts/run.py --config configs/medical.yaml

# with the LLM reranker
python scripts/run.py --config configs/rarebench.yaml --rerank
```

## 5. Evaluate

```bash
python scripts/evaluate.py --config configs/rarebench.yaml --run

python scripts/evaluate.py --config configs/medical.yaml --run
```

RareBench is scored with the LLM judge and reports Hit@k and Median Rank. The medical
config answers each question from the retrieved passages and grades the answer, reporting
accuracy.
