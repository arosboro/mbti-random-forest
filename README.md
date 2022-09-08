# mbti-random-forest

## With Term Frequency-Inverse Document Frequency Index trial 3
Calculate Vec<Vec<64>> TF-IDF dense matrix for x_train,y_train.  Use SVC model for Generic algorithm.
Saving samples...
Saving x and y matrices...
106067 x samples
106067 y labels
16 unique labels
Saving dictionary...
Dictionary size: 183130
Creating tf from corpus...
tf: 71 minutes
Creating idf from corpus...

## Bigger Better Data Set trial 2
Removed stopwords, underfitting generic model, but ensemble of 4 classifier models is more accurate than trial 1
    Saving samples...
    Saving x and y matrices...
    106067 samples
    16 unique labels
    Saving dictionary...
    Dictionary size: 210146
    Tallying...
    count ISTJ: 1243
    count ISFJ: 650
    count INFJ: 14963
    count INTJ: 22427
    count ISTP: 3424
    count ISFP: 875
    count INFP: 12134
    count INTP: 24961
    count ESTP: 1986
    count ESFP: 360
    count ENFP: 6167
    count ENTP: 11725
    count ESTJ: 482
    count ESFJ: 181
    count ENFJ: 1534
    count ENTJ: 2955
    Tally of [IE, NS, TF, JP]: IE
    80677 samples for I
    25390 samples for E
    Tally of [IE, NS, TF, JP]: NS
    96866 samples for N
    9201 samples for S
    Tally of [IE, NS, TF, JP]: TF
    69203 samples for T
    36864 samples for F
    Tally of [IE, NS, TF, JP]: JP
    44435 samples for J
    61632 samples for P
    Generating generic model
    Training...
    RandomForestClassifierParameters { criterion: Gini, max_depth: None, min_samples_leaf: 1, min_samples_split: 2, n_trees: 100, m: None, keep_samples: false, seed: 0 }
    RandomForestClassifierParameters { criterion: Gini, max_depth: Some(8), min_samples_leaf: 128, min_samples_split: 256, n_trees: 16, m: Some(64), keep_samples: true, seed: 42 }
    Serializing random forest...
    Iteration: 0
    Random Forest accuracy: 0.2342431527836704
    MSE: 979.3300334700419
    Loading random forest...
    Validation of serialized model...
    Metrics about model.
    Accuracy: 0.2342431527836704
    MSE: 979.3300334700419
    Training [IE, NS, TF, JP]: IE
    Training...
    RandomForestClassifierParameters { criterion: Gini, max_depth: None, min_samples_leaf: 1, min_samples_split: 2, n_trees: 100, m: None, keep_samples: false, seed: 0 }
    RandomForestClassifierParameters { criterion: Gini, max_depth: Some(8), min_samples_leaf: 128, min_samples_split: 256, n_trees: 16, m: Some(64), keep_samples: true, seed: 42 }
    Serializing random forest...
    Iteration: 0
    Random Forest accuracy: 0.7598170932918493
    MSE: 0.24018290670815065
    Loading random forest...
    Validation of serialized model...
    Metrics about model.
    Accuracy: 0.7598170932918493
    MSE: 0.24018290670815065
    Training [IE, NS, TF, JP]: NS
    Training...
    RandomForestClassifierParameters { criterion: Gini, max_depth: None, min_samples_leaf: 1, min_samples_split: 2, n_trees: 100, m: None, keep_samples: false, seed: 0 }
    RandomForestClassifierParameters { criterion: Gini, max_depth: Some(8), min_samples_leaf: 128, min_samples_split: 256, n_trees: 16, m: Some(64), keep_samples: true, seed: 42 }
    Serializing random forest...
    Iteration: 0
    Random Forest accuracy: 0.9112336774619337
    MSE: 0.08876632253806628
    Loading random forest...
    Validation of serialized model...
    Metrics about model.
    Accuracy: 0.9112336774619337
    MSE: 0.08876632253806628
    Training [IE, NS, TF, JP]: TF
    Training...
    RandomForestClassifierParameters { criterion: Gini, max_depth: None, min_samples_leaf: 1, min_samples_split: 2, n_trees: 100, m: None, keep_samples: false, seed: 0 }
    RandomForestClassifierParameters { criterion: Gini, max_depth: Some(8), min_samples_leaf: 128, min_samples_split: 256, n_trees: 16, m: Some(64), keep_samples: true, seed: 42 }
    Serializing random forest...
    Iteration: 0
    Random Forest accuracy: 0.6511573091971904
    MSE: 0.3488426908028096
    Loading random forest...
    Validation of serialized model...
    Metrics about model.
    Accuracy: 0.6511573091971904
    MSE: 0.3488426908028096
    Training [IE, NS, TF, JP]: JP
    Training...
    RandomForestClassifierParameters { criterion: Gini, max_depth: None, min_samples_leaf: 1, min_samples_split: 2, n_trees: 100, m: None, keep_samples: false, seed: 0 }
    RandomForestClassifierParameters { criterion: Gini, max_depth: Some(8), min_samples_leaf: 128, min_samples_split: 256, n_trees: 16, m: Some(64), keep_samples: true, seed: 42 }
    Serializing random forest...
    Iteration: 0
    Random Forest accuracy: 0.5760147079620987
    MSE: 0.4239852920379013
    Loading random forest...
    Validation of serialized model...
    Metrics about model.
    Accuracy: 0.5760147079620987
    MSE: 0.4239852920379013

## Small Data Set trial 1
Didn't remove stopwords... less accurate because underfitting data.

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