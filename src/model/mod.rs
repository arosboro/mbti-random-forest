use std::path::Path;

use smartcore::ensemble::random_forest_classifier::{
    RandomForestClassifier, RandomForestClassifierParameters,
};
use smartcore::metrics::{accuracy, mean_squared_error, roc_auc_score};
use smartcore::model_selection::{cross_validate, KFold};
use smartcore::svm::svc::{SVCParameters, SVC};
use smartcore::svm::Kernels;
use smartcore::tree::decision_tree_classifier::SplitCriterion;
use smartcore::{linalg::naive::dense_matrix::DenseMatrix, svm::LinearKernel};

use rust_ml_helpers::serialize::write_file_truncate;

const PATH_ML_RF_ALL_MODEL: &str = "mbti_rf__ALL.model";

pub fn train(corpus: &Vec<Vec<f64>>, classifiers: &Vec<u8>, member_id: &str) {
    println!("Training... {}", member_id);
    println!("x shape: ({}, {})", corpus.len(), corpus[0].len());
    println!("y shape: ({}, {})", classifiers.len(), 1);

    // let sub_x = corpus[0..5000].to_vec();
    // let sub_y = classifiers[0..5000].to_vec();
    let sub_x = corpus;
    let sub_y = classifiers;

    let x: DenseMatrix<f64> = DenseMatrix::new(
        sub_x.len(),
        sub_x[0].len(),
        sub_x.clone().into_iter().flatten().collect(),
    );
    // These are our target class labels
    let y: Vec<f64> = sub_y.into_iter().map(|x| *x as f64).collect();
    // Split bag into training/test (80%/20%)
    // let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true);

    let rf_ml_wrapper = |x: &DenseMatrix<f64>,
                         y: &Vec<f64>,
                         parameters: RandomForestClassifierParameters| {
        RandomForestClassifier::fit(x, y, parameters).and_then(|rf| {
            unsafe {
                static mut ITERATIONS: u32 = 0;
                ITERATIONS += 1;
                if ITERATIONS % 10 == 0 {
                    ITERATIONS = 0;
                    println!("Serializing random forest...");
                    let path = Path::new(PATH_ML_RF_ALL_MODEL);
                    write_file_truncate(&path, &bincode::serialize(&rf).unwrap())
                        .expect(format!("Can not persist random_forest {}", member_id).as_str());
                }
                println!("Iteration: {}", ITERATIONS);
            }
            // Evaluate
            let y_hat_rf: Vec<f64> = rf.predict(x).unwrap();
            let m1 = accuracy(y, &y_hat_rf);
            let m2 = mean_squared_error(y, &y_hat_rf);

            println!("Random forest accuracy: {}, MSE: {}", m1, m2);

            Ok(rf)
        })
    };

    let svm_ml_wrapper =
        |x: &DenseMatrix<f64>,
         y: &Vec<f64>,
         parameters: SVCParameters<f64, DenseMatrix<f64>, LinearKernel>| {
            SVC::fit(x, y, parameters).and_then(|svm| {
                unsafe {
                    static mut ITERATIONS: u32 = 0;
                    ITERATIONS += 1;
                    if ITERATIONS % 10 == 0 {
                        ITERATIONS = 0;
                        println!("Serializing support vector machine...");
                        let filename = format!("./mbti_svm__{}.model", member_id);
                        let path_model: &Path = Path::new(&filename);
                        write_file_truncate(path_model, &bincode::serialize(&svm).unwrap())
                            .expect("Failed to write samples");
                    }
                    println!("Iteration: {}", ITERATIONS);
                }
                // Evaluate
                let y_hat_svm: Vec<f64> = svm.predict(x).unwrap();
                let m1 = accuracy(y, &y_hat_svm);
                let m2 = mean_squared_error(y, &y_hat_svm);
                let m3 = roc_auc_score(y, &y_hat_svm);

                println!("SVM accuracy: {}, MSE: {}, AUC SVM: {}", m1, m2, m3);

                Ok(svm)
            })
        };

    // Parameters
    const DEFAULT_RF_PARAMS: RandomForestClassifierParameters = RandomForestClassifierParameters {
        criterion: SplitCriterion::Gini,
        max_depth: None,
        min_samples_leaf: 1,
        min_samples_split: 2,
        n_trees: 100,
        m: Option::None,
        keep_samples: false,
        seed: 0,
    };

    const TWEAKED_RF_PARAMS: RandomForestClassifierParameters = RandomForestClassifierParameters {
        /// Split criteria to use when building a tree. See [Decision Tree Classifier](../../tree/decision_tree_classifier/index.html)
        criterion: SplitCriterion::Gini,
        /// Tree max depth. See [Decision Tree Classifier](../../tree/decision_tree_classifier/index.html)
        max_depth: None,
        /// The minimum number of samples required to be at a leaf node. See [Decision Tree Classifier](../../tree/decision_tree_classifier/index.html)
        min_samples_leaf: 4,
        /// The minimum number of samples required to split an internal node. See [Decision Tree Classifier](../../tree/decision_tree_classifier/index.html)
        min_samples_split: 8,
        /// The number of trees in the forest.
        n_trees: 256,
        /// Number of random sample of predictors to use as split candidates.
        m: Option::None,
        /// Whether to keep samples used for tree generation. This is required for OOB prediction.
        keep_samples: false,
        /// Seed used for bootstrap sampling and feature selection for each tree.
        seed: 42u64,
    };

    let default_svm_params: SVCParameters<f64, DenseMatrix<f64>, LinearKernel> =
        SVCParameters::default()
            .with_epoch(2)
            .with_c(1.0)
            .with_tol(0.001)
            .with_kernel(Kernels::linear());

    let tweaked_svm_params: SVCParameters<f64, DenseMatrix<f64>, LinearKernel> =
        SVCParameters::default()
            .with_epoch(2)
            .with_c(2000.0)
            .with_tol(0.0001)
            .with_kernel(Kernels::linear());

    // Random Forest
    let splits = 10;
    println!("{:.0}", (corpus.len() / splits) as f64);
    if member_id == "ALL" {
        println!("{:?}", DEFAULT_RF_PARAMS);
        println!("{:?}", TWEAKED_RF_PARAMS);
        let results = cross_validate(
            rf_ml_wrapper,
            &x,
            &y,
            TWEAKED_RF_PARAMS,
            KFold::default().with_n_splits(splits),
            accuracy,
        )
        .unwrap();
        println!(
            "Test score: {}, training score: {}",
            results.mean_test_score(),
            results.mean_train_score()
        );
    }
    // SVM Ensemble
    else {
        println!("{:?}", default_svm_params);
        println!("{:?}", tweaked_svm_params);
        let results = cross_validate(
            svm_ml_wrapper,
            &x,
            &y,
            default_svm_params,
            KFold::default().with_n_splits(splits),
            accuracy,
        )
        .unwrap();
        println!(
            "Test score: {}, training score: {}",
            results.mean_test_score(),
            results.mean_train_score()
        );
    }
}
