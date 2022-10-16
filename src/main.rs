use nalgebra::DMatrix;
use rust_ml_helpers::serialize::load_bytes;
use smartcore::ensemble::random_forest_classifier::RandomForestClassifier;
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::metrics::{accuracy, mean_squared_error};
use smartcore::model_selection::train_test_split;
use smartcore::svm::svc::SVC;
use smartcore::svm::LinearKernel;
use std::path::Path;

use crate::csv_loader::{load_data, Sample};
use crate::ensemble::build_sets;
use crate::model::train;
use crate::myers_briggs::{indicator, MBTI};
use crate::normalize::normalize;
use crate::report::tally;
use crate::visualize::{create_charts, visualization_json};

pub mod csv_loader;
pub mod ensemble;
pub mod model;
pub mod myers_briggs;
pub mod normalize;
pub mod report;
pub mod visualize;

const PATH_MBTI_RANDOM_FOREST_ALL_MODEL: &str = "mbti_rf__ALL.model";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let training_set: Vec<Sample> = load_data();
    let (_data_x, _data_y) = normalize(training_set);
    let (data_x, data_y) = create_charts();
    visualization_json();
    let train_test_length: usize = (data_x.len() as f64 * 0.9) as usize;
    let validate_length: usize = (data_x.len() as f64 * 0.1) as usize;
    let (corpus, classifiers) = {
        let mut corpus: Vec<Vec<f64>> = Vec::new();
        let mut classifiers: Vec<u8> = Vec::new();
        for i in 0..train_test_length {
            corpus.push(data_x.get(i).unwrap().clone());
            classifiers.push(data_y.get(i).unwrap().clone());
        }
        (corpus, classifiers)
    };
    let (validate_x, validate_y) = {
        let mut validate_x: Vec<Vec<f64>> = Vec::new();
        let mut validate_y: Vec<u8> = Vec::new();
        for i in train_test_length..train_test_length + validate_length {
            validate_x.push(data_x.get(i).unwrap().clone());
            validate_y.push(data_y.get(i).unwrap().clone());
        }
        (validate_x, validate_y)
    };
    let trees = ["IE", "NS", "TF", "JP"];
    tally(&classifiers);
    // Build sets for an ensemble of models
    let (ie_corpus, ie_classifiers, ie_labels) = build_sets(
        &validate_x,
        &validate_y,
        indicator::mb_flag::I,
        indicator::mb_flag::E,
    );
    let (ns_corpus, ns_classifiers, ns_labels) = build_sets(
        &validate_x,
        &validate_y,
        indicator::mb_flag::N,
        indicator::mb_flag::S,
    );
    let (tf_corpus, tf_classifiers, tf_labels) = build_sets(
        &validate_x,
        &validate_y,
        indicator::mb_flag::T,
        indicator::mb_flag::F,
    );
    let (jp_corpus, jp_classifiers, jp_labels) = build_sets(
        &validate_x,
        &validate_y,
        indicator::mb_flag::J,
        indicator::mb_flag::P,
    );
    let ensemble = [
        (ie_corpus, ie_classifiers, ie_labels),
        (ns_corpus, ns_classifiers, ns_labels),
        (tf_corpus, tf_classifiers, tf_labels),
        (jp_corpus, jp_classifiers, jp_labels),
    ];
    // Build sets of an ensemble of models having a single classifier
    // Train models
    for i in 0..ensemble.len() {
        println! {"Tally of [IE, NS, TF, JP]: {}", trees[i]};
        println! {"{} samples for {}", ensemble[i].1.iter().filter(|&n| *n == 0u8).count(), trees[i].chars().nth(0).unwrap()};
        println! {"{} samples for {}", ensemble[i].1.iter().filter(|&n| *n == 1u8).count(), trees[i].chars().nth(1).unwrap()};
    }
    let model_rf_all_path = Path::new(PATH_MBTI_RANDOM_FOREST_ALL_MODEL);
    if !model_rf_all_path.exists() {
        println!("Generating generic model");
        train(&corpus, &classifiers, "ALL");
    } else {
        println!("Generic models already exists");
        // Evaluate
        // let y_hat_rf: Vec<f64> = rf.predict(&x_test).unwrap();
        // println!("Random Forest accuracy: {}", accuracy(&y_test, &y_hat_rf));
        // println!("MSE: {}", mean_squared_error(&y_test, &y_hat_rf));
    }
    // Load the rf Model
    println!("Loading random forest...");
    let rf: RandomForestClassifier<f64> = {
        let path = Path::new("mbti_rf__ALL.model");
        bincode::deserialize(&load_bytes(path)).expect("Can not deserialize the model")
    };
    let mut models: Vec<SVC<f64, DenseMatrix<f64>, LinearKernel>> = Vec::new();
    for i in 0..ensemble.len() {
        let _final_model = (i + 1) * 3;
        let filename = format!("./mbti_svm__{}.model", trees[i]);

        let path_model = Path::new(&filename);
        if !path_model.exists() {
            println! {"Training SVM model for {}", trees[i]};
            train(&ensemble[i].0, &ensemble[i].1, &trees[i]);
            let svm: SVC<f64, DenseMatrix<f64>, LinearKernel> = {
                bincode::deserialize(&load_bytes(path_model))
                    .expect("Can not deserialize the model")
            };
            models.push(svm);
        } else {
            println!("Loading svm {} model...", trees[i]);
            let svm: SVC<f64, DenseMatrix<f64>, LinearKernel> = {
                bincode::deserialize(&load_bytes(path_model))
                    .expect("Can not deserialize the model")
            };
            models.push(svm);
        }
    }
    // Get predictions for ensemble
    let mut ensemble_y_test: [Vec<f64>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let mut ensemble_pred: [Vec<f64>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let types: [[u8; 2]; 4] = [
        [indicator::mb_flag::I, indicator::mb_flag::E],
        [indicator::mb_flag::N, indicator::mb_flag::S],
        [indicator::mb_flag::T, indicator::mb_flag::F],
        [indicator::mb_flag::J, indicator::mb_flag::P],
    ];
    for (i, model) in models.iter().enumerate() {
        let x = DenseMatrix::from_2d_vec(&ensemble[i].0);
        let y = ensemble[i]
            .1
            .iter()
            .map(|x| *x as f64)
            .collect::<Vec<f64>>();
        let (_x_train, x_test, _y_train, y_test) = train_test_split(&x, &y, 0.2, true);
        ensemble_pred[i] = model.predict(&x_test).unwrap();
        ensemble_y_test[i] = y_test;
        println!(
            "{} accuracy: {}",
            trees[i],
            accuracy(&ensemble_y_test[i], &ensemble_pred[i])
        );
        println!(
            "MSE: {}",
            mean_squared_error(&ensemble_y_test[i], &ensemble_pred[i])
        );
    }
    let mut svm_ensemble_y_test: Vec<f64> = Vec::new();
    let mut svm_ensemble_y_pred: Vec<f64> = Vec::new();
    assert!(ensemble_pred.len() == 4);
    for i in 0..ensemble_pred[0].len() {
        let mut mbti: u8 = 0u8;
        for j in 0..ensemble.len() {
            let prediction = ensemble_pred[j].get(i).unwrap();
            // trees = ["IE", "NS", "TF", "JP"]; (Defined above)
            let tree = trees[j];
            let leaf_a: char = tree.chars().nth(0).unwrap();
            let leaf_b: char = tree.chars().nth(1).unwrap();
            let low: bool = *prediction == 0f64;
            let flag: u8 = {
                let flag: u8;
                if low {
                    flag = match leaf_a {
                        'I' => indicator::mb_flag::I,
                        'N' => indicator::mb_flag::N,
                        'T' => indicator::mb_flag::T,
                        'J' => indicator::mb_flag::J,
                        _ => 0u8,
                    };
                } else {
                    flag = match leaf_b {
                        'E' => indicator::mb_flag::E,
                        'S' => indicator::mb_flag::S,
                        'F' => indicator::mb_flag::F,
                        'P' => indicator::mb_flag::P,
                        _ => 0u8,
                    };
                }
                flag
            };
            mbti |= flag;
        }
        svm_ensemble_y_pred.push(mbti as f64);
        let ie_idx = *ensemble_y_test[0].get(i).unwrap() as usize;
        let ns_idx = *ensemble_y_test[1].get(i).unwrap() as usize;
        let tf_idx = *ensemble_y_test[2].get(i).unwrap() as usize;
        let jp_idx = *ensemble_y_test[3].get(i).unwrap() as usize;
        let ie = types[0][ie_idx];
        let ns = types[1][ns_idx];
        let tf = types[2][tf_idx];
        let jp = types[3][jp_idx];
        let test_mbti = ie ^ ns ^ tf ^ jp;
        svm_ensemble_y_test.push(test_mbti as f64);
    }

    // Evaluate
    let sample_report = |y: &Vec<f64>, y_hat: &Vec<f64>| {
        let indicator_accuracy = |a: &String, b: &String| -> f64 {
            let mut correct: f64 = 0f64;
            for i in 0..a.len() {
                if a.chars().nth(i).unwrap() == b.chars().nth(i).unwrap() {
                    correct += 1f64;
                }
            }
            correct / a.len() as f64
        };
        let variance = |y: &Vec<f64>, y_hat: &Vec<f64>| -> f64 {
            let mut acc: Vec<f64> = Vec::new();
            for i in 0..y_hat.len() {
                let predicted: String = MBTI {
                    indicator: y_hat[i] as u8,
                }
                .to_string();
                let actual: String = MBTI {
                    indicator: y[i] as u8,
                }
                .to_string();
                let variance = indicator_accuracy(&predicted, &actual);
                acc.push(variance);
            }
            DMatrix::from_vec(acc.len(), 1, acc).mean()
        };
        let mean_variance = variance(y_hat, y);
        println!("Prediction, Actual, Variance, Mean Variance, Correct");
        for i in 0..25 {
            let predicted: String = MBTI {
                indicator: y_hat[i] as u8,
            }
            .to_string();
            let actual: String = MBTI {
                indicator: y[i] as u8,
            }
            .to_string();
            let diff = {
                let mut acc: String = "    ".to_string();
                for (i, char) in predicted.chars().enumerate() {
                    if char != actual.chars().nth(i).unwrap() {
                        acc.push(char);
                    } else {
                        acc.push(' ');
                    }
                }
                acc
            };
            let variance = indicator_accuracy(&predicted, &actual);
            println!(
                "{},       {}, {},     {:.2},     {:.2}           {}",
                predicted,
                actual,
                diff,
                variance,
                mean_variance,
                y_hat[i] == y[i]
            );
        }
    };
    // Load data for Random Forest test predictions
    let x: DenseMatrix<f64> = DenseMatrix::from_2d_vec(&corpus);
    // These are our target class labels
    let y: Vec<f64> = classifiers.into_iter().map(|x| x as f64).collect();
    // Split bag into training/test (80%/20%)
    let (_x_train, x_test, _y_train, y_test) = train_test_split(&x, &y, 0.2, true);
    println!(
        "Generic Random Forest accuracy: {}",
        accuracy(&y_test, &rf.predict(&x_test).unwrap())
    );
    sample_report(&y_test, &rf.predict(&x_test).unwrap());
    println!(
        "Ensemble Support Vector Machine accuracy: {}",
        accuracy(&svm_ensemble_y_test, &svm_ensemble_y_pred)
    );
    sample_report(&svm_ensemble_y_test, &svm_ensemble_y_pred);
    Ok(())
}
