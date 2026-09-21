# Where LLMs, RAG and classical ML sit

My own summary of how the pieces fit together, written while building the PDF Q&A RAG app.

## The map

```
                        ARTIFICIAL INTELLIGENCE
                                  |
                          MACHINE LEARNING
                     (learn patterns from data)
                        |                    |
           CLASSICAL ML                    DEEP LEARNING
     (structured / tabular data)       (unstructured data)
     linear + logistic regression       neural nets, many layers
     decision trees, XGBoost,           trained by backpropagation
     SVM, k-means                              |
     -> churn, fraud, credit,           +------+----------+
        forecasting                    CNN    RNN    TRANSFORMER
     STILL THE DEFAULT FOR TABLES    images  seqs   (self-attention)
                                                          |
                                            +-------------+------------+
                                      ENCODER-style              DECODER-style
                                   embedding models                  LLMs
                              (all-MiniLM in my app)        (Gemini, GPT, Claude)
                                   text -> vectors           tokens -> next token
                                                                     |
                                                      multimodal: images become
                                                      patches (ViT), not CNN
```

## Giving an LLM knowledge it was not trained on

| Approach | How it works | When to use |
| --- | --- | --- |
| Context stuffing | Put the whole file in the prompt. No chunking, no search | Small input: one image, a couple of pages |
| RAG | Embed the question, search a vector store, put the top-k chunks in the prompt | Large or changing content, many documents, citations needed, per-user access control |
| Fine-tuning | Train the model's own weights further | Consistent style, tone or output format. Not a reliable way to add facts |

## Why classical ML is not obsolete

For tabular data, gradient-boosted trees usually beat deep learning, and they are far cheaper and explainable.

| Problem | Typical model | Why |
| --- | --- | --- |
| Customer churn | XGBoost / logistic regression | Tabular, needs explainability |
| Fraud detection | Gradient boosting + rules | Millisecond latency, regulator-facing |
| Sales forecasting | Regression, ARIMA, boosting | Small data, seasonality |
| Credit scoring | Logistic regression | Decision must be explainable |
| Contract summarisation | LLM | Language |
| Q&A over documents | LLM + RAG | Language, private data |
| Defect detection in photos | CNN / vision transformer | Images, cheap hardware |

Deep learning is for unstructured data; classical ML owns structured data. LLMs extended the toolbox, they did not replace it.

## Notes worth remembering

- The final layer of an LLM picks the next token with softmax, which is multi-class logistic regression. The old ideas are the building blocks.
- Self-attention is what transformers added: every token can look at every other token, and training parallelises on GPUs.
- Multimodal LLMs cut an image into patches and embed each patch as a token (Vision Transformer). They do not use a CNN.
- An embedding model (encoder, bi-encoder) reads text and outputs vectors. An LLM (decoder) generates text. Both are transformers.

## How my project maps onto this

```
PDF -> chunk -> [encoder transformer] -> vectors -> Chroma
                                                      |
question -> [same encoder] -> vector ------> search --+
                                                      |
                                          top-3 chunks + question
                                                      |
                                        [decoder transformer = Gemini]
                                                      |
                                          answer with citations
```

## My interview answer

Machine learning splits into classical ML for structured, tabular data (regression, decision trees, gradient
boosting, which still win on cost, latency and explainability for problems like churn or fraud) and deep learning
for unstructured data like text, images and audio. An LLM sits in the deep learning branch: it is a decoder-only
transformer trained to predict the next token, and self-attention is what lets it handle long-range context. My CNN
project and my RAG project are two applications of the same foundation: convolutions for images, attention for
language. Because a model only knows its training data, there are three ways to give it new knowledge: put the
content directly in the prompt, which works only for small inputs; fine-tune it, which changes style and behaviour
rather than reliably adding facts; or use RAG, which retrieves the relevant pieces at query time. I built the RAG
path: I chunk a PDF, embed the chunks with an encoder model, store them in a vector database, retrieve the top
matches by cosine distance for each question, and pass those to Gemini with a prompt that forces it to answer only
from that context and cite the chunks. That keeps answers grounded, updatable and checkable, which is what
production systems need.

## Follow-up questions to be ready for

1. Why not fine-tune instead of RAG?
2. How do you know your retrieval is any good? (Answer properly once Ragas evaluation is in — week 3)
3. Context windows are huge now, is RAG dead?
4. How would you do RAG over scanned documents with no text layer?
