# Phase 4: Evaluation and Experimental Results

## 1. Evaluation Methodology

The retrieval and classification components of Saarthi were evaluated using labelled question datasets. The evaluation focused on retrieval quality, ranking effectiveness, classification accuracy, and multilingual retrieval performance.

The primary retrieval metric was **Normalized Discounted Cumulative Gain (nDCG)**, which measures the quality of ranked results while accounting for the relevance grades assigned to retrieved verses. Precision, Recall, Mean Reciprocal Rank (MRR), and Mean Average Precision (MAP) were also used in multilingual evaluation.

The experiments covered:

- BM25 parameter tuning and holdout evaluation.
- Independent comparison of TF-IDF and BM25.
- Evaluation of Naive Bayes, Rocchio, and k-Nearest Neighbours classifiers.
- Multilingual retrieval across English, Hindi, and Hinglish.

## 2. BM25 Parameter Tuning

BM25 was evaluated using a grid search over different values of its two primary parameters:

- **k1:** Controls the influence of term frequency.
- **b:** Controls document-length normalization.

The following parameter combinations were tested:

- k1: 0.8, 1.2, 1.5, 1.8, 2.0
- b: 0.25, 0.5, 0.75, 1.0

This produced 20 parameter combinations, evaluated on 20 labelled tuning questions using nDCG@20.

### Results

The best-performing tested configuration was:

| Parameter | Selected value |
|---|---:|
| k1 | 1.2 |
| b | 1.0 |
| nDCG@20 | 0.06501 |
| Questions evaluated | 20 |
| Empty result sets | 0 |

The next-best configuration was k1 = 0.8 and b = 1.0, with an nDCG@20 of 0.064798.

The selected configuration was subsequently used for BM25 holdout and independent retrieval evaluation.

**Observation:** The best tested value of b was at the upper boundary of the search grid. Therefore, this result identifies the best configuration among the tested combinations, rather than establishing a global optimum.

The tuning results are saved in `evaluation/results/bm25_tuning.csv`.

## 3. BM25 Holdout Evaluation

After selecting the best BM25 parameters during tuning, the configuration \(k_1 = 1.2\) and \(b = 1.0\) was evaluated on a separate holdout set containing 12 labelled questions (q055–q066).

### Results

| Metric | Result |
|---|---:|
| Questions evaluated | 12 |
| nDCG@20 | 0.0186 |
| Empty result sets | 0 |

Although the system returned results for all 12 questions, the average nDCG@20 was relatively low. Two questions (q055 and q066) received non-zero nDCG scores, while the remaining questions received zero scores.

**Observation:** The holdout performance was considerably lower than the tuning performance of 0.06501. This suggests that the selected configuration did not generalize equally well to the holdout questions. The holdout set was kept separate from parameter selection to avoid tuning directly to its results.

## 4. Independent Retrieval Comparison

TF-IDF and BM25 were compared using 34 labelled questions (q067–q100). Both methods were evaluated using nDCG@20, with BM25 using the parameters selected during tuning (\(k_1 = 1.2\), \(b = 1.0\)).

### Results

| Retrieval method | Questions | Empty results | nDCG@20 |
|---|---:|---:|---:|
| TF-IDF | 34 | 0 | 0.0675 |
| BM25 | 34 | 0 | 0.0643 |

### Observations

- TF-IDF achieved a slightly higher nDCG@20 than BM25.
- The difference between the two methods was 0.0032, indicating a relatively small performance gap.
- Neither method produced empty result sets.
- Both methods were evaluated on the same labelled questions, allowing a direct comparison under the same evaluation conditions.

**Conclusion:** TF-IDF performed marginally better than BM25 on this independent labelled evaluation subset. However, the difference is small and should not be interpreted as evidence of a statistically significant advantage.

## 5. Classifier Evaluation

The classification component was evaluated using 68 labelled English questions across 11 themes. Leave-one-out cross-validation was used: each question was held out once while the remaining 67 questions were used for training. The TF-IDF vectorizer was fitted separately within each fold using only the training data.

The evaluated methods included Naive Bayes, Rocchio, k-Nearest Neighbours (kNN), and two TF-IDF similarity baselines.

### Results

| Classifier | Configuration | Correct | Accuracy | Macro F1 |
|---|---|---:|---:|---:|
| TF-IDF baseline | Mean similarity | 35/68 | 51.47% | 0.48 |
| Rocchio | Centroid similarity | 34/68 | 50.00% | 0.44 |
| kNN | k = 1 | 28/68 | 41.18% | 0.38 |
| TF-IDF baseline | Maximum similarity | 28/68 | 41.18% | 0.41 |
| kNN | k = 3 | 26/68 | 38.24% | 0.35 |
| kNN | k = 5 | 26/68 | 38.24% | 0.33 |
| kNN | k = 7 | 25/68 | 36.76% | 0.30 |
| Naive Bayes | MultinomialNB | 22/68 | 32.35% | 0.26 |

### Observations

- The TF-IDF mean-similarity baseline achieved the highest accuracy (51.47%) and macro F1 score (0.48).
- Rocchio was the second-best method, with 50% accuracy and a macro F1 score of 0.44.
- kNN performance decreased as the number of neighbours increased from 1 to 7.
- Multinomial Naive Bayes achieved the lowest accuracy among the evaluated methods.
- Some themes, particularly Devotion, were difficult to classify correctly across the evaluated methods.

**Conclusion:** The existing TF-IDF mean-similarity approach performed best on this dataset, with Rocchio providing a competitive alternative. Naive Bayes and kNN were also evaluated as additional classification approaches. These results are preliminary and should be interpreted in the context of the dataset size and leave-one-out evaluation setup.

## 6. Multilingual Retrieval Evaluation

Multilingual retrieval was evaluated across English, Hindi, and Hinglish. Three retrieval strategies were compared:

- **Direct:** Retrieval using the original query.
- **Translate English:** Retrieval using the English-translated query.
- **Hindi-side:** Retrieval using the Hindi-side strategy implemented in the system.

Each language-strategy combination was evaluated on 34 questions. The metrics included Precision@5, Recall@10, Mean Reciprocal Rank (MRR), Mean Average Precision (MAP), and nDCG@10.

### Results

| Language | Strategy | P@5 | R@10 | MRR | MAP | nDCG@10 |
|---|---|---:|---:|---:|---:|---:|
| English | Direct | 0.171 | 0.261 | 0.362 | 0.134 | 0.218 |
| English | Translate English | 0.171 | 0.261 | 0.377 | 0.136 | 0.225 |
| English | Hindi-side | 0.088 | 0.136 | 0.219 | 0.057 | 0.131 |
| Hindi | Direct | 0.100 | 0.141 | 0.205 | 0.057 | 0.111 |
| Hindi | Translate English | 0.171 | 0.265 | 0.377 | 0.146 | 0.236 |
| Hindi | Hindi-side | 0.100 | 0.147 | 0.220 | 0.060 | 0.134 |
| Hinglish | Direct | 0.029 | 0.056 | 0.077 | 0.018 | 0.045 |
| Hinglish | Translate English | 0.165 | 0.260 | 0.322 | 0.138 | 0.222 |
| Hinglish | Hindi-side | 0.082 | 0.121 | 0.214 | 0.053 | 0.117 |

### Observations

- For English queries, direct retrieval and English translation produced similar precision and recall. Translation achieved slightly higher MRR, MAP, and nDCG@10.
- For Hindi queries, the English-translation strategy performed best across all five metrics, achieving an nDCG@10 of 0.236.
- For Hinglish queries, direct retrieval performed poorly, with an nDCG@10 of 0.045.
- Translating Hinglish queries into English substantially improved retrieval, achieving an nDCG@10 of 0.222.
- The Hindi-side strategy performed better than direct retrieval for Hinglish, but remained below the English-translation strategy.

**Conclusion:** English translation improved retrieval effectiveness for Hindi and Hinglish queries in these experiments. The results highlight the importance of query normalization and translation in supporting multilingual access to Saarthi. Hinglish direct retrieval remains an area for improvement.

## 7. Discussion

The experiments provide an overall comparison of Saarthi's retrieval and classification components.

### 7.1 Retrieval Performance

The independent retrieval evaluation showed that TF-IDF achieved a slightly higher nDCG@20 (0.0675) than BM25 (0.0643). BM25's best tuning configuration was \(k_1 = 1.2\) and \(b = 1.0\), but its holdout nDCG@20 was lower than its tuning score.

These results suggest that retrieval performance varies across question sets and that parameter tuning on a limited dataset does not necessarily guarantee better performance on unseen questions.

### 7.2 Classification Performance

The TF-IDF mean-similarity baseline achieved the best classification accuracy (51.47%) and macro F1 score (0.48). Rocchio performed comparably, while kNN and Multinomial Naive Bayes achieved lower scores.

The results suggest that simpler similarity-based approaches can be competitive with the additional classifiers on the current labelled dataset.

### 7.3 Multilingual Performance

The multilingual evaluation showed that translating Hindi and Hinglish queries into English improved retrieval effectiveness compared with direct retrieval.

In particular, Hinglish direct retrieval had the weakest results, while the translation strategy produced a substantial improvement. This indicates that multilingual query processing is an important component of the system.

## 8. Limitations and Future Improvements

The evaluation has several limitations:

1. **Dataset size:** The labelled datasets are relatively small, which limits the generalizability of the reported results.
2. **Retrieval performance:** The absolute nDCG scores indicate that there is considerable scope to improve the ranking and relevance of retrieved passages.
3. **Parameter tuning:** The best BM25 configuration was selected from a finite grid. The selected value of \(b = 1.0\) was at the boundary of the tested range.
4. **Classification accuracy:** The classifier results indicate that theme classification remains challenging, particularly for some themes with limited or difficult examples.
5. **Multilingual processing:** Direct Hinglish retrieval performed poorly, showing a need for better normalization, transliteration, and query-processing techniques.
6. **Evaluation coverage:** The results reflect the labelled questions and evaluation procedures used in these experiments. Further testing on larger and more diverse datasets is required.

Future improvements could include expanding the labelled question dataset, improving relevance annotations, experimenting with additional retrieval configurations, refining multilingual query normalization, and evaluating more advanced ranking and classification approaches.

## 9. Conclusion

Phase 4 evaluated Saarthi's retrieval, classification, and multilingual search components using labelled datasets and quantitative metrics.

The experiments identified TF-IDF mean similarity as the strongest evaluated classification baseline and showed that TF-IDF slightly outperformed the tuned BM25 configuration on the independent labelled retrieval subset. The multilingual experiments demonstrated that English translation improved retrieval effectiveness for Hindi and Hinglish queries.

Although the results reveal areas requiring further improvement, they provide a quantitative baseline for comparing the implemented methods and guiding future development of Saarthi.

The evaluation results and supporting scripts are maintained in the `evaluation/` directory, with generated result files stored in `evaluation/results/`.

