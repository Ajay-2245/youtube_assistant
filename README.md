Improvements over time:

1) Added Langsmith tracing

2) Identified that translation task is consuming lot of time if video is 1hr long or above.

        Approach  was to solve it by chunking first and send raw non-translated chunks asynchronously to LLM. got an 8x improvement in latency.


Preparing the evaluation dataset : 

1) Prepared eval dataset using RAGAS, only one video is considered and 30 QA pairs are generated.



# Retrieval Evaluation Metrics — Summary

Common setup for all metrics below: a query, the ranked top-k documents your retriever returned, and a ground-truth set of "relevant" documents for that query.

---

## 1. Precision@k

`(relevant docs in top-k) / k`

- **Measures:** how much of what you retrieved is actually useful (signal vs. noise).
- **Production use:** moderate, rarely used alone. In RAG, low precision means junk context reaches the LLM, increasing hallucination risk and token cost.

## 2. Recall@k

`(relevant docs in top-k) / (total relevant docs that exist)`

- **Measures:** whether you found *all* the needed info, anywhere in top-k, regardless of position.
- **Production use:** **the most common retrieval metric in RAG systems specifically.**
- **Why:** an LLM generator reads the whole context window and doesn't care about order within it — the real failure mode is "was the answer-bearing chunk retrieved at all," which is exactly recall.
- ⭐ **For a RAG chatbot, this is the primary metric to optimize.**

## 3. Mean Reciprocal Rank (MRR)

`1 / (rank of first relevant doc)`, averaged across queries.

- **Measures:** how quickly the *first* relevant result appears — rewards ranking.
- **Production use:** common in classic search/QA with one right answer (autocomplete, lookup) where position matters a lot. Less critical for RAG-with-LLM-generator (order within context matters less), but cheap and useful as a secondary signal.

## 4. Mean Average Precision (MAP)

Average of precision values computed at each rank where a relevant doc appears, then averaged across queries.

- **Measures:** precision *and* ranking quality combined, rewards putting all relevant docs early.
- **Production use:** historical gold standard in classic IR (web search, TREC) with many relevant docs per query. Assumes **binary relevance** — can't express "doc A is more relevant than doc B." Modern RAG pipelines tend to reach for nDCG instead when ranking-awareness matters.

## 5. Normalized Discounted Cumulative Gain (nDCG@k)

```
DCG@k  = Σ (rel_i / log2(i + 1))   for i = 1 to k
IDCG@k = DCG@k computed on the ideal (best-possible) ordering of the same docs
nDCG@k = DCG@k / IDCG@k
```

- **Measures:** both *whether* you found relevant docs and *how well-ranked* they are, using **graded relevance** (not just binary relevant/not).
- **Production use:** industry standard in large-scale ranking (web search, e-commerce, recsys) where relevance is naturally graded and a human scans the ranked list.
- **Motivation over MAP/Recall:** binary relevance forces an arbitrary line between "fully relevant" and "not" — nDCG lets you score partial relevance (e.g. 0–3 grades) and rewards rankings that put the highest-graded docs first.

### Worked example
5 retrieved docs, graded relevance (0–3):

| Rank | Grade |
|------|-------|
| 1 | 3 |
| 2 | 0 |
| 3 | 2 |
| 4 | 0 |
| 5 | 1 |

DCG@5 = 3/log2(2) + 0 + 2/log2(4) + 0 + 1/log2(6) = 3 + 1 + 0.387 = **4.387**
IDCG@5 (ideal order [3,2,1,0,0]) = 3 + 1.262 + 0.5 = **4.762**
nDCG@5 = 4.387 / 4.762 ≈ **0.921**

Compare to Recall@5 = 1.0 for the same example (all 3 relevant docs were retrieved) — nDCG correctly reveals the ranking wasn't ideal even though recall says "perfect."

### ⚠️ Important clarification (common misreading)
"A badly-ranked highly-relevant doc hurts more than a badly-ranked somewhat-relevant one" is **only meaningful when comparing different orderings of the *same* document set for a single query** — not across queries with different ground truths, and not by comparing raw DCG contributions directly.

- If a query has only **one** relevant doc, the grade cancels out of nDCG entirely — burying a grade-3 doc at rank 2 gives the *same* nDCG as burying a grade-1 doc at rank 2 (both ≈0.631 in a 2-doc example). No "hurts more" effect exists here.
- The effect only shows up with **multiple graded relevant docs**, comparing swaps within the same set:
  - Displacing the *highest*-graded doc from rank 1 → nDCG ≈ 0.9225
  - Displacing a *lower*-graded doc among the remaining ranks → nDCG ≈ 0.9725
  - Same document set, same IDCG — only the internal ordering changed.
- **Why:** nDCG normalizes by each query's own IDCG, which erases any basis for comparing raw contributions *across* different document sets/queries.

### Caveat for this project
Since ground truth here (`reference_contexts` from the ragas eval set) is **binary** (a chunk either generated the question or didn't), nDCG collapses to something close to rank-weighted recall unless relevance is manually graded — in which case Recall@k and MRR already capture most of the signal for less effort.

---

## Bottom line

| Metric | Best used for | Priority for this project |
|---|---|---|
| Recall@k | "Did we find the needed info at all?" | ⭐ Primary |
| Precision@k | "How much noise is in the context?" | Strong secondary |
| MRR | Ranking quality, single-answer lookup | Diagnostic |
| MAP | Ranking quality, many relevant docs, binary relevance | Rarely needed here |
| nDCG@k | Ranking quality with graded relevance | Only valuable if you manually grade partial relevance |

**Decide k** by sweeping k = 1, 3, 5, 10, 15, 20 and plotting recall/precision/nDCG at each — pick the point where recall plateaus without precision dropping too far (the "knee" of the curve). Check where your current hardcoded `k=5` lands on that curve, and consider computing metrics separately for single-hop vs. multi-hop questions since they have different numbers of ground-truth relevant chunks (1 vs. 2).