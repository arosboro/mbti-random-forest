use nalgebra::DMatrix;
use rust_decimal::prelude::*;
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use crate::csv_loader::{Lemma, Lemmas, Sample};
use crate::myers_briggs::indicator;
use rust_ml_helpers::normalize::*;
use rust_ml_helpers::serialize::*;

pub type Dictionary = HashMap<String, f64>;
pub type Feature = f64;
pub type Features = Vec<Feature>;

const PATH_DICTIONARY: &str = "./dictionary.bincode";
const PATH_DOCUMENT_FREQUENCY: &str = "./document_frequency.bincode";
const PATH_PERSONALITY_FREQUENCY: &str = "./personality_frequency.bincode";
const PATH_OVERALL_FREQUENCY: &str = "./overall_frequency.bincode";
const PATH_TF_MATRIX: &str = "./tf_matrix.bincode";
const PATH_IDF_MATRIX: &str = "./idf_matrix.bincode";
const PATH_TFIDF_MATRIX: &str = "./tfidf_matrix.bincode";
const PATH_TERM_CORPUS: &str = "./term_corpus.bincode";
const PATH_TERM_CLASSIFIERS: &str = "./term_classifiers.bincode";

const INDICATORS: [u8; 16] = [
    indicator::ESFJ,
    indicator::ESFP,
    indicator::ESTJ,
    indicator::ESTP,
    indicator::ENFJ,
    indicator::ENFP,
    indicator::ENTJ,
    indicator::ENTP,
    indicator::ISFJ,
    indicator::ISFP,
    indicator::ISTJ,
    indicator::ISTP,
    indicator::INFJ,
    indicator::INFP,
    indicator::INTJ,
    indicator::INTP,
];

pub fn normalize(training_set: Vec<Sample>) -> (Vec<Features>, Vec<u8>) {
    let path_fx = Path::new("./corpus.bincode");
    let path_fy = Path::new("./classifiers.bincode");
    if path_fx.exists() || path_fy.exists() {
        println!("Loading x and y matrices...");
        let corpus: Vec<Vec<f64>> = bincode::deserialize(&load_bytes(path_fx)).unwrap();
        let classifiers: Vec<u8> = bincode::deserialize(&load_bytes(path_fy)).unwrap();
        (corpus, classifiers)
    } else {
        println!("Saving x and y matrices...");
        let mut x_set: Vec<Lemmas> = Vec::new();
        let y_set: Vec<u8> = training_set.iter().map(|x| x.label.indicator).collect();
        for sample in training_set.iter() {
            x_set.push(sample.lemmas.clone());
        }
        println!("{} x samples", x_set.len());
        println!("{} y labels", y_set.len());
        let term_corpus: DMatrix<Lemma> =
            DMatrix::from_fn(x_set.len(), x_set[0].len(), |i, j| x_set[i][j].to_owned());
        let term_classifiers: DMatrix<u8> = DMatrix::from_fn(y_set.len(), 1, |i, _| y_set[i]);

        // Deterimine unique labels
        let mut unique_labels: Vec<String> = Vec::new();
        for label in y_set.iter() {
            let mbti = *label;
            if !unique_labels.contains(&mbti.to_string()) {
                unique_labels.push(mbti.to_string());
            }
        }
        println!("{} unique labels", unique_labels.len());

        // Determine unique words, initial round of statistics (standard frequency based)
        let (dictionary, df_index, overall_frequency, personality_frequency) =
            process_corpus(&x_set, &y_set);
        println!("dictionary size: {}", dictionary.len());
        println!("df_index size: {}", df_index.len());
        println!("overall_frequency size: {}", overall_frequency.len());
        println!(
            "personality_frequency size: {:?}",
            personality_frequency.into_iter().fold(
                Vec::new(),
                |mut acc: Vec<usize>, x: HashMap<String, f64>| {
                    acc.push(x.len());
                    acc
                }
            )
        );

        // Create a dense matrix of term frequencies.
        let tf_matrix: Vec<Vec<f64>> = {
            let path = Path::new(PATH_TF_MATRIX);
            if path.exists() {
                println!("Loading tf_matrix...");
                let tf_matrix: Vec<Vec<f64>> = bincode::deserialize(&load_bytes(path)).unwrap();
                tf_matrix
            } else {
                println!("Creating a dense matrix of term frequencies...");
                let start = Instant::now();
                let tf_matrix: DMatrix<f64> =
                    DMatrix::from_fn(term_corpus.nrows(), term_corpus.ncols(), |i, j| -> f64 {
                        tf(
                            term_corpus.slice((i, 0), (1, term_corpus.ncols())),
                            &term_corpus[(i, j)],
                        )
                    });
                println!("tf_matrix: {} seconds", start.elapsed().as_secs());
                println!(
                    "We obtained a {}x{} matrix",
                    tf_matrix.nrows(),
                    tf_matrix.ncols()
                );
                let mut corpus_tf: Vec<Vec<f64>> = Vec::new();
                for i in 0..tf_matrix.nrows() {
                    let mut row: Vec<f64> = Vec::new();
                    for j in 0..tf_matrix.ncols() {
                        row.push(tf_matrix[(i, j)]);
                    }
                    corpus_tf.push(row);
                }
                println!("Saving tf_matrix...");
                let tf_matrix_bytes = bincode::serialize(&corpus_tf).unwrap();
                write_file_truncate(&path, &tf_matrix_bytes).expect("Failed to write tf_matrix");
                corpus_tf
            }
        };

        // Create a dense matrix of idf values.
        let idf_matrix: Vec<Vec<f64>> = {
            let path = Path::new(PATH_IDF_MATRIX);
            if path.exists() {
                println!("Loading idf_matrix...");
                let idf_matrix: Vec<Vec<f64>> = bincode::deserialize(&load_bytes(path)).unwrap();
                idf_matrix
            } else {
                println!("Creating a dense matrix of idf values...");
                let start = Instant::now();
                let idf_matrix: DMatrix<f64> =
                    DMatrix::from_fn(term_corpus.nrows(), term_corpus.ncols(), |i, j| -> f64 {
                        idf(df_index[&term_corpus[(i, j)]], term_corpus.nrows())
                    });
                println!("idf_matrix: {} seconds", start.elapsed().as_secs());
                println!(
                    "We obtained a {}x{} matrix",
                    idf_matrix.nrows(),
                    idf_matrix.ncols()
                );
                let mut corpus_idf: Vec<Vec<f64>> = Vec::new();
                // Convert idf to a Vec<Vec<f64>>.
                for i in 0..idf_matrix.nrows() {
                    let mut row: Vec<f64> = Vec::new();
                    for j in 0..idf_matrix.ncols() {
                        row.push(idf_matrix[(i, j)]);
                    }
                    corpus_idf.push(row);
                }
                println!("Saving idf_matrix...");
                let idf_matrix_bytes = bincode::serialize(&corpus_idf).unwrap();
                write_file_truncate(&path, &idf_matrix_bytes).expect("Failed to write idf_matrix");
                corpus_idf
            }
        };

        // Finally, create the tf-idf matrix by multiplying.
        let tf_idf: Vec<Features> = {
            let path = Path::new(PATH_TFIDF_MATRIX);
            if path.exists() {
                println!("Loading tf_idf...");
                let tf_idf: Vec<Vec<f64>> = bincode::deserialize(&load_bytes(path)).unwrap();
                tf_idf
            } else {
                println!("Creating the tf-idf matrix by multiplying...");
                let start = Instant::now();
                let tf_idf: DMatrix<Feature> = DMatrix::from_fn(
                    term_corpus.nrows(),
                    term_corpus.ncols(),
                    |i, j| -> Feature {
                        let score: f64 = (tf_matrix[i][j] * idf_matrix[i][j]) as f64;
                        score
                    },
                );
                // TODO: Normalizing makes the values too tiny.
                let max = tf_idf.max();
                let min = tf_idf.min();
                let tf_idf_normal: DMatrix<f64> =
                    DMatrix::from_fn(tf_idf.nrows(), tf_idf.ncols(), |i, j| tf_idf[(i, j)] / max);
                println!(
                    "tf_idf max: {}, tf_idf_normal max: {}",
                    max,
                    tf_idf_normal.max()
                );
                println!("tf_idf: {} seconds", start.elapsed().as_secs());
                // Convert tf_idf to a Vec<Vec<f64>>.
                println!("max: {}", max);
                println!("min: {}", min);
                println!("normal max: {}", tf_idf_normal.max());
                println!("normal min: {}", tf_idf_normal.min());
                let mut corpus_tf_idf: Vec<Vec<f64>> = Vec::new();
                for x in 0..tf_idf.nrows() {
                    let mut row: Vec<f64> = Vec::new();
                    for y in 0..tf_idf_normal.ncols() {
                        let val = tf_idf_normal[(x, y)];
                        let decimal_val: Decimal = Decimal::from_f64(val).unwrap();
                        row.push(decimal_val.round_dp(4).to_f64().unwrap());
                    }
                    corpus_tf_idf.push(row);
                }
                corpus_tf_idf
            }
        };

        serialize_term_corpus(&term_corpus);
        serialize_term_classifiers(&term_classifiers);

        let classifiers: Vec<u8> = term_classifiers.iter().map(|y| *y).collect();

        (tf_idf, classifiers)
    }
}

pub fn process_corpus(
    x_set: &Vec<Lemmas>,
    y_set: &Vec<u8>,
) -> (Dictionary, Dictionary, Dictionary, Vec<Dictionary>) {
    let path_dictionary: &Path = Path::new(PATH_DICTIONARY);
    let path_df_index: &Path = Path::new(PATH_DOCUMENT_FREQUENCY);
    let path_personality_frequency: &Path = Path::new(PATH_PERSONALITY_FREQUENCY);
    let path_overall_frequency: &Path = Path::new(PATH_OVERALL_FREQUENCY);
    if path_dictionary.exists()
        && path_df_index.exists()
        && path_personality_frequency.exists()
        && path_overall_frequency.exists()
    {
        load_processed_corpus()
    } else {
        // Saving dictionary
        println!("Saving all statistical mappings...");
        let mut dictionary: Dictionary = HashMap::new();
        let mut overall_frequency: Dictionary = HashMap::new();
        let mut personality_frequency: Vec<Dictionary> = vec![HashMap::new(); 16];
        let mut df_major: Dictionary = HashMap::new();
        for (i, post) in x_set.iter().enumerate() {
            // Build dictionary containing unique ids for each term
            build_dictionary(&mut dictionary, &post);
            // Build document frequency counting each term in a single document.
            // Generate overall frequency each term is used in the corpus.
            let df_minor: HashMap<String, f64> =
                build_document_frequency_overall_frequency(&mut overall_frequency, &post);
            // Count frequency at which each classifier uses each term
            build_classifier_frequency(
                &mut personality_frequency,
                &INDICATORS.to_vec(),
                &y_set[i],
                &df_minor,
            );

            // Build document frequency counting each term once per document over all documents
            df_minor.iter().for_each(|(term, _)| {
                *df_major.entry(term.to_string()).or_insert(0.0) += 1.0;
            });
        }
        // Serialize the dictionary.
        println!("Saving dictionary...");
        let dictionary_bytes = bincode::serialize(&dictionary).unwrap();
        write_file_truncate(&path_dictionary, &dictionary_bytes)
            .expect("Failed to write dictionary");
        // Serialize the df_index.
        println!("Saving df_index...");
        let df_major_bytes = bincode::serialize(&df_major).unwrap();
        write_file_truncate(&path_df_index, &df_major_bytes).expect("Failed to write df_index");
        // Serialize the overall_frequency.
        println!("Saving overall_frequency...");
        let overall_freq_bytes = bincode::serialize(&overall_frequency).unwrap();
        write_file_truncate(&path_overall_frequency, &overall_freq_bytes)
            .expect("Failed to write overall_frequency");
        // Serialize the personality_frequency.
        println!("Saving personality_frequency...");
        let personality_frequency_bytes = bincode::serialize(&personality_frequency).unwrap();
        write_file_truncate(&path_personality_frequency, &personality_frequency_bytes)
            .expect("Failed to write personality_frequency");

        (
            dictionary,
            df_major, // df_index
            overall_frequency,
            personality_frequency,
        )
    }
}

pub fn load_processed_corpus() -> (Dictionary, Dictionary, Dictionary, Vec<Dictionary>) {
    let path_dictionary: &Path = Path::new(PATH_DICTIONARY);
    let path_df_index: &Path = Path::new(PATH_DOCUMENT_FREQUENCY);
    let path_personality_frequency: &Path = Path::new(PATH_PERSONALITY_FREQUENCY);
    let path_overall_frequency: &Path = Path::new(PATH_OVERALL_FREQUENCY);
    println!("Loading dictionary...");
    let dictionary: Dictionary = bincode::deserialize(&load_bytes(path_dictionary)).unwrap();
    println!("Loading df_index...");
    let df_major: Dictionary = bincode::deserialize(&load_bytes(path_df_index)).unwrap();
    println!("Loading overall_frequency...");
    let overall_frequency: Dictionary =
        bincode::deserialize(&load_bytes(path_overall_frequency)).unwrap();
    println!("Loading personality_frequency...");
    let personality_frequency: Vec<Dictionary> =
        bincode::deserialize(&load_bytes(path_personality_frequency)).unwrap();

    (
        dictionary,
        df_major, // df_index
        overall_frequency,
        personality_frequency,
    )
}

pub fn serialize_term_corpus(term_corpus: &DMatrix<Lemma>) {
    let path_term_corpus: &Path = Path::new(PATH_TERM_CORPUS);
    println!("Saving term_corpus...");
    let term_corpus_bytes = bincode::serialize(&term_corpus).unwrap();
    write_file_truncate(&path_term_corpus, &term_corpus_bytes)
        .expect("Failed to write term_corpus");
}

pub fn serialize_term_classifiers(term_classifiers: &DMatrix<u8>) {
    let path_term_classifiers: &Path = Path::new(PATH_TERM_CLASSIFIERS);
    println!("Saving term_classifiers...");
    let term_classifiers_bytes = bincode::serialize(&term_classifiers).unwrap();
    write_file_truncate(&path_term_classifiers, &term_classifiers_bytes)
        .expect("Failed to write term_classifiers");
}

pub fn load_term_corpus() -> DMatrix<Lemma> {
    let path_term_corpus: &Path = Path::new(PATH_TERM_CORPUS);
    println!("Loading term_corpus...");
    bincode::deserialize(&load_bytes(path_term_corpus)).unwrap()
}

pub fn load_classifiers() -> DMatrix<u8> {
    let path_term_classifiers: &Path = Path::new(PATH_TERM_CLASSIFIERS);
    println!("Loading term_classifiers...");
    bincode::deserialize(&load_bytes(path_term_classifiers)).unwrap()
}
