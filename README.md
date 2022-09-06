# mbti-random-forest

```zsh
Loading samples...
Loading x and y matrices...
Tallying...
count ISTJ: 9705
count ISFJ: 7946
count INFJ: 70676
count INTJ: 51594
count ISTP: 16116
count ISFP: 12630
count INFP: 87708
count INTP: 62057
count ESTP: 4257
count ESFP: 2176
count ENFP: 32238
count ENTP: 33231
count ESTJ: 1895
count ESFJ: 2008
count ENFJ: 9172
count ENTJ: 11086
Tally of [IE, NS, TF, JP]: IE
318432 samples for I
96063 samples for E
Tally of [IE, NS, TF, JP]: NS
357762 samples for N
56733 samples for S
Tally of [IE, NS, TF, JP]: TF
189941 samples for T
224554 samples for F
Tally of [IE, NS, TF, JP]: JP
164082 samples for J
250413 samples for P
Training [IE, NS, TF, JP]: IE
Training...
RandomForestClassifierParameters { criterion: Gini, max_depth: None, min_samples_leaf: 1, min_samples_split: 2, n_trees: 100, m: None, keep_samples: false, seed: 0 }
RandomForestClassifierParameters { criterion: Gini, max_depth: Some(14), min_samples_leaf: 256, min_samples_split: 128, n_trees: 4, m: Some(64), keep_samples: true, seed: 42 }
Serializing random forest...
Iteration: 0
Random Forest accuracy: 0.770829563685931
MSE: 0.22917043631406891
Loading random forest...
Validation of serialized model...
Metrics about model.
Accuracy: 0.770829563685931
MSE: 0.22917043631406891
Training [IE, NS, TF, JP]: NS
Training...
RandomForestClassifierParameters { criterion: Gini, max_depth: None, min_samples_leaf: 1, min_samples_split: 2, n_trees: 100, m: None, keep_samples: false, seed: 0 }
RandomForestClassifierParameters { criterion: Gini, max_depth: Some(14), min_samples_leaf: 256, min_samples_split: 128, n_trees: 4, m: Some(64), keep_samples: true, seed: 42 }
Serializing random forest...
Iteration: 0
Random Forest accuracy: 0.8620128107697318
MSE: 0.13798718923026815
Loading random forest...
Validation of serialized model...
Metrics about model.
Accuracy: 0.8620128107697318
MSE: 0.13798718923026815
Training [IE, NS, TF, JP]: TF
Training...
RandomForestClassifierParameters { criterion: Gini, max_depth: None, min_samples_leaf: 1, min_samples_split: 2, n_trees: 100, m: None, keep_samples: false, seed: 0 }
RandomForestClassifierParameters { criterion: Gini, max_depth: Some(14), min_samples_leaf: 256, min_samples_split: 128, n_trees: 4, m: Some(64), keep_samples: true, seed: 42 }
Serializing random forest...
Iteration: 0
Random Forest accuracy: 0.5425035283899685
MSE: 0.4574964716100315
Loading random forest...
Validation of serialized model...
Metrics about model.
Accuracy: 0.5425035283899685
MSE: 0.4574964716100315
Training [IE, NS, TF, JP]: JP
Training...
RandomForestClassifierParameters { criterion: Gini, max_depth: None, min_samples_leaf: 1, min_samples_split: 2, n_trees: 100, m: None, keep_samples: false, seed: 0 }
RandomForestClassifierParameters { criterion: Gini, max_depth: Some(14), min_samples_leaf: 256, min_samples_split: 128, n_trees: 4, m: Some(64), keep_samples: true, seed: 42 }
Serializing random forest...
Iteration: 0
Random Forest accuracy: 0.6012859021218592
MSE: 0.3987140978781409
Loading random forest...
Validation of serialized model...
Metrics about model.
Accuracy: 0.6012859021218592
MSE: 0.3987140978781409
```