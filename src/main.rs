use csv::Error;
use image::ImageBuffer;
use rand::seq::SliceRandom;
use rand::thread_rng;
// use rand::Rng;
// use rust_bert::pipelines::pos_tagging::POSConfig;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use smartcore::svm::svc::{SVCParameters, SVC};
use smartcore::svm::{Kernels, LinearKernel};
use smartcore::tree::decision_tree_classifier::SplitCriterion;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;
// DenseMatrix wrapper around Vec
use smartcore::linalg::naive::dense_matrix::*;
// Random Forest
use smartcore::ensemble::random_forest_classifier::{
    RandomForestClassifier, RandomForestClassifierParameters,
};
// Model performance
use nalgebra::{DMatrix, DMatrixSlice};
use regex::Regex;
// use rust_bert::pipelines::pos_tagging::{POSModel, POSTag};
use rust_decimal::prelude::*;
use rust_stemmers::{Algorithm, Stemmer};
use smartcore::metrics::{accuracy, mean_squared_error, roc_auc_score};
use smartcore::model_selection::{cross_validate, train_test_split, KFold};
use stopwords::{Language, Spark, Stopwords};
use vtext::tokenize::*;
extern crate image;

// Stop words borrowed from analysis by github.com/riskmakov/MBTI-predictor
const ENNEAGRAM_TERMS: [&str; 23] = [
    "1w2",
    "2w1",
    "2w3",
    "3w2",
    "3w4",
    "4w3",
    "4w5",
    "6w5",
    "6w7",
    "7w6",
    "7w8",
    "8w7",
    "8w9",
    "9w8",
    "so",
    "sp",
    "sx",
    "enneagram",
    "ennegram",
    "socionics",
    "tritype",
    "7s",
    "8s",
];
const IMAGE_TERMS: [&str; 14] = [
    "gif",
    "giphy",
    "image",
    "img",
    "imgur",
    "jpeg",
    "jpg",
    "JPG",
    "photobucket",
    "php",
    "player_embedded",
    "png",
    "staticflickr",
    "tinypic",
];
const INVALID_WORDS: [&str; 12] = [
    "im",
    "mbti",
    "functions",
    "myers",
    "briggs",
    "types",
    "type",
    "personality",
    "16personalities",
    "16",
    "personalitycafe",
    "tapatalk",
];
const MBTI_TERMS: [&str; 16] = [
    "intj", "intp", "entj", "entp", "infj", "infp", "enfj", "enfp", "istj", "isfj", "estj", "esfj",
    "istp", "isfp", "estp", "esfp",
];
const MBTI_PARTS: [&str; 35] = [
    "es", "fj", "fs", "nt", "nf", "st", "sf", "sj", "nts", "nfs", "sts", "sfs", "sjs", "enxp",
    "esxp", "exfj", "exfp", "extp", "inxj", "ixfp", "ixtp", "ixtj", "ixfj", "xnfp", "xntp", "xnfj",
    "xnfp", "xstj", "xstp", "nfp", "ntp", "ntj", "nfp", "sfj", "sps",
];

#[derive(Debug, Deserialize)]
struct Row {
    r#type: String,
    posts: String,
}

pub mod indicator {
    pub mod mb_flag {
        pub const I: u8 = 0b10000000;
        pub const E: u8 = 0b01000000;
        pub const S: u8 = 0b00100000;
        pub const N: u8 = 0b00010000;
        pub const T: u8 = 0b00001000;
        pub const F: u8 = 0b00000100;
        pub const J: u8 = 0b00000010;
        pub const P: u8 = 0b00000001;
    }
    pub const ISTJ: u8 = mb_flag::I ^ mb_flag::S ^ mb_flag::T ^ mb_flag::J;
    pub const ISFJ: u8 = mb_flag::I ^ mb_flag::S ^ mb_flag::F ^ mb_flag::J;
    pub const INFJ: u8 = mb_flag::I ^ mb_flag::N ^ mb_flag::F ^ mb_flag::J;
    pub const INTJ: u8 = mb_flag::I ^ mb_flag::N ^ mb_flag::T ^ mb_flag::J;
    pub const ISTP: u8 = mb_flag::I ^ mb_flag::S ^ mb_flag::T ^ mb_flag::P;
    pub const ISFP: u8 = mb_flag::I ^ mb_flag::S ^ mb_flag::F ^ mb_flag::P;
    pub const INFP: u8 = mb_flag::I ^ mb_flag::N ^ mb_flag::F ^ mb_flag::P;
    pub const INTP: u8 = mb_flag::I ^ mb_flag::N ^ mb_flag::T ^ mb_flag::P;
    pub const ESTP: u8 = mb_flag::E ^ mb_flag::S ^ mb_flag::T ^ mb_flag::P;
    pub const ESFP: u8 = mb_flag::E ^ mb_flag::S ^ mb_flag::F ^ mb_flag::P;
    pub const ENFP: u8 = mb_flag::E ^ mb_flag::N ^ mb_flag::F ^ mb_flag::P;
    pub const ENTP: u8 = mb_flag::E ^ mb_flag::N ^ mb_flag::T ^ mb_flag::P;
    pub const ESTJ: u8 = mb_flag::E ^ mb_flag::S ^ mb_flag::T ^ mb_flag::J;
    pub const ESFJ: u8 = mb_flag::E ^ mb_flag::S ^ mb_flag::F ^ mb_flag::J;
    pub const ENFJ: u8 = mb_flag::E ^ mb_flag::N ^ mb_flag::F ^ mb_flag::J;
    pub const ENTJ: u8 = mb_flag::E ^ mb_flag::N ^ mb_flag::T ^ mb_flag::J;
}

#[derive(Debug, Copy, Serialize, Deserialize, Clone)]
struct MBTI {
    indicator: u8,
}

impl MBTI {
    pub fn to_string(&self) -> String {
        match self.indicator {
            indicator::ISTJ => "ISTJ".to_owned(),
            indicator::ISFJ => "ISFJ".to_owned(),
            indicator::INFJ => "INFJ".to_owned(),
            indicator::INTJ => "INTJ".to_owned(),
            indicator::ISTP => "ISTP".to_owned(),
            indicator::ISFP => "ISFP".to_owned(),
            indicator::INFP => "INFP".to_owned(),
            indicator::INTP => "INTP".to_owned(),
            indicator::ESTP => "ESTP".to_owned(),
            indicator::ESFP => "ESFP".to_owned(),
            indicator::ENFP => "ENFP".to_owned(),
            indicator::ENTP => "ENTP".to_owned(),
            indicator::ESTJ => "ESTJ".to_owned(),
            indicator::ESFJ => "ESFJ".to_owned(),
            indicator::ENFJ => "ENFJ".to_owned(),
            indicator::ENTJ => "ENTJ".to_owned(),
            _ => "Unknown".to_owned(),
        }
    }

    pub fn from_string(input: &str) -> MBTI {
        let mbti = input.chars().fold(0b00000000, |acc, c| match c {
            'I' => acc ^ indicator::mb_flag::I,
            'E' => acc ^ indicator::mb_flag::E,
            'S' => acc ^ indicator::mb_flag::S,
            'N' => acc ^ indicator::mb_flag::N,
            'T' => acc ^ indicator::mb_flag::T,
            'F' => acc ^ indicator::mb_flag::F,
            'J' => acc ^ indicator::mb_flag::J,
            'P' => acc ^ indicator::mb_flag::P,
            _ => panic!("Invalid MBTI"),
        });
        MBTI { indicator: mbti }
    }
}

type Post = Vec<String>;
type Lemmas = Vec<String>;
type Dictionary = HashMap<String, f64>;
// type POSDictionary = HashMap<String, String>;
type Features = f64;

#[derive(Debug, Serialize, Deserialize, Clone)]
struct Sample {
    label: MBTI,
    lemmas: Lemmas,
}

fn load_bytes(path: &Path) -> Vec<u8> {
    let mut buf = Vec::new();
    File::open(path)
        .unwrap()
        .read_to_end(&mut buf)
        .expect(&format!("Unable to read file {}", path.to_str().unwrap()));
    buf
}

// Remove regular expression targets from a post and return the result
// The expressions are:
// Regex::new(r"https?://[a-zA-Z0-9_%./]+\??(([a-z0-9%]+=[a-z0-9%]+&?)+)?").unwrap(),
// Regex::new(r"[^a-zA-Z0-9 ]").unwrap(),
// Regex::new(r"\s+").unwrap(),
fn cleanup(post: &str, expressions: &[Regex; 3]) -> String {
    let mut acc: String = post.to_owned();
    for expression in expressions.iter() {
        // replace with a space if there are more than one spaces
        let rep = if expression.as_str() == r"\s+" {
            " "
        // replace with an empty string if the replacement was meant to be removed
        } else {
            ""
        };
        // apply the regular expression with replace all and use rep which we set above
        acc = expression.replace_all(&acc, rep).to_string();
    }
    acc.to_owned()
}

// accumulate a list of words to remove from the dataset
fn get_stopwords() -> HashSet<&'static str> {
    let mut stopwords: HashSet<&str> = Spark::stopwords(Language::English)
        .unwrap()
        .iter()
        .map(|&s| s)
        .collect();
    for &x in ENNEAGRAM_TERMS.iter() {
        stopwords.insert(x);
    }
    for &x in IMAGE_TERMS.iter() {
        stopwords.insert(x);
    }
    for &x in MBTI_PARTS.iter() {
        stopwords.insert(x);
    }
    for &x in MBTI_TERMS.iter() {
        stopwords.insert(x);
    }
    for &x in INVALID_WORDS.iter() {
        stopwords.insert(x);
    }
    return stopwords;
}

// resolve bases of words to their root form
fn lemmatize(lemmas: Lemmas) -> Vec<String> {
    let mut acc = Vec::new();
    for token in lemmas {
        let stemmer = Stemmer::create(Algorithm::English);
        let lemma = stemmer.stem(&token);
        acc.push(lemma.to_string());
    }
    acc
}

// turn a string post into a vector of lemmas which represent
// that string as individual lemmatized terms
fn tokenize(post: &str, expressions: &[Regex; 3]) -> Vec<String> {
    let clean = cleanup(post.to_lowercase().as_str(), expressions);
    let stopwords = get_stopwords();
    let tokenizer = VTextTokenizerParams::default().lang("en").build().unwrap();
    let mut tokens: Vec<&str> = tokenizer.tokenize(&clean).collect();
    // a token could be empty so, retain ones with length greater than one
    tokens.retain(|&x| x.trim().len() > 0);
    // a token could be in stopwords so, retain ones that are not in stopwords
    tokens.retain(|&token| !stopwords.contains(token));
    let clean_tokens: Lemmas = tokens.iter().map(|&x| x.trim().to_string()).collect();
    lemmatize(clean_tokens)
}

// The data isn't balanced, try to oversample underrepresented classes
// fn oversample(samples: Vec<Sample>) -> Vec<Sample> {
//     let mut oversampled = Vec::new();
//     // counts contains the number of each classifier occuring in the dataset
//     let mut counts = HashMap::new();
//     for s in samples.iter() {
//         let count = counts.entry(s.label.indicator).or_insert(0);
//         *count += 1;
//     }
//     let max = counts.values().max().unwrap();
//     for (i, s) in samples.iter().enumerate() {
//         let count = counts.get(&s.label.indicator).unwrap();
//         let ratio = max / count;
//         oversampled.push(s.clone());
//         for _ in 0..ratio {
//             let mut new_sample = s.clone();
//             let mut new_features = Vec::new();
//             let mut index = 0;
//             let mut l: usize = if i > 0 { i - 1 } else { 0 };
//             let mut r: usize = i + 1;
//             let mut neighbors: [usize; 5] = [0; 5];
//             // This will simulate selecting random terms from the 5 nearest neighbors which have the same label.
//             while new_features.len() < s.lemmas.len() && (l > 0 || r < samples.len()) {
//                 while index < neighbors.len()
//                     && (l > 0 || r < samples.len())
//                     && index < neighbors.len()
//                 {
//                     // go left.
//                     if l < i && l > 0 {
//                         if samples[l].label.indicator == new_sample.label.indicator {
//                             neighbors[index] = l;
//                             index += 1;
//                         }
//                         l = if l > 1 { l - 1 } else { 0 };
//                     }
//                     if index < neighbors.len() {
//                         // go right.
//                         if r > i && r < samples.len() {
//                             if samples[r].label.indicator == new_sample.label.indicator {
//                                 neighbors[index] = r;
//                                 index += 1;
//                             }
//                             r += 1;
//                         }
//                     }
//                 }
//                 if index == neighbors.len() {
//                     for _ in 0..s.lemmas.len() {
//                         // random number between 0..neighbors.len()
//                         let n = rand::thread_rng().gen_range(0..neighbors.len());
//                         // one of the 5 nearest neighbors.
//                         let k = neighbors[n];
//                         // random number between 0..s.lemmas.len()
//                         let j = rand::thread_rng().gen_range(0..s.lemmas.len());
//                         new_features.push(samples[k].lemmas[j].clone());
//                     }
//                     break;
//                 }
//                 if index >= neighbors.len() {
//                     break;
//                 }
//             }
//             new_sample.lemmas = new_features;
//             oversampled.push(new_sample);
//         }
//     }
//     oversampled
// }

fn load_data() -> Vec<Sample> {
    let csv_target = Path::new("./mbti_1.csv");
    let mut samples: Vec<Sample> = {
        let path: &Path = Path::new("./samples.bincode");
        if path.exists() {
            println!("Loading samples...");
            bincode::deserialize(&load_bytes(path)).unwrap()
        } else {
            println!("Saving samples...");
            const BALANCED_SAMPLES: bool = true;
            let mut samples: Vec<Sample> = Vec::new();
            let mut reader = csv::Reader::from_path(csv_target).unwrap();
            let expressions = [
                Regex::new(r"https?://[a-zA-Z0-9_%./]+\??(([a-z0-9%]+=[a-z0-9%]+&?)+)?").unwrap(),
                Regex::new(r"[^a-zA-Z0-9 ]").unwrap(),
                Regex::new(r"\s+").unwrap(),
            ];
            let mut counters: HashMap<u8, usize> = HashMap::new();
            for row in reader.deserialize::<Row>() {
                match row {
                    Ok(row) => {
                        // Choose the first POST_SIZE lemmas for a composite post.
                        const MANUAL_POST_SIZE: usize = 32;
                        const MANUAL_VS_AVG: bool = true;
                        if MANUAL_VS_AVG {
                            let mut lemma_group: Vec<String> = Vec::new();
                            for post in row.posts.split("|||").collect::<Vec<&str>>() {
                                let lemmas = tokenize(post, &expressions);
                                if lemmas.len() > 0 {
                                    lemma_group.extend(lemmas);
                                }
                            }
                            let mut lemma_post: Vec<String> = Vec::new();
                            lemma_group.iter().enumerate().for_each(|(i, lemma)| {
                                lemma_post.push(lemma.clone());
                                if (i + 1) % MANUAL_POST_SIZE == 0 {
                                    let lemmas = lemma_post.clone();
                                    samples.push(Sample {
                                        lemmas,
                                        label: MBTI::from_string(&row.r#type),
                                    });
                                    counters
                                        .entry(MBTI::from_string(&row.r#type).indicator)
                                        .and_modify(|e| *e += 1)
                                        .or_insert(1);
                                    lemma_post.clear();
                                }
                            });
                        } else {
                            let sample_row: Vec<Sample> = row
                                .posts
                                .split("|||")
                                .map(|post| {
                                    counters
                                        .entry(MBTI::from_string(&row.r#type).indicator)
                                        .and_modify(|e| *e += 1)
                                        .or_insert(1);
                                    Sample {
                                        lemmas: tokenize(post, &expressions),
                                        label: MBTI::from_string(&row.r#type),
                                    }
                                })
                                .collect();
                            samples.extend(sample_row);
                        }
                    }
                    Err(e) => println!("Error: {}", e),
                }
            }

            // Ensure the same number of features and classifiers exist in each category.
            let mut balanced_samples: Vec<Sample> = Vec::new();
            let min_count = *counters.values().min().unwrap();
            let mut entry_counters: HashMap<u8, usize> = HashMap::new();
            for sample in samples.iter() {
                let count = *entry_counters.get(&sample.label.indicator).unwrap_or(&0);
                if count < min_count {
                    entry_counters
                        .entry(sample.label.indicator)
                        .and_modify(|e| *e += 1)
                        .or_insert(1);
                    balanced_samples.push(sample.clone());
                }
            }
            println! {"{} samples per classifier.", min_count};
            // Save samples
            let f = std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(path);
            let samples_bytes = bincode::serialize(if BALANCED_SAMPLES {
                &balanced_samples
            } else {
                &samples
            })
            .unwrap();
            f.and_then(|mut f| f.write_all(&samples_bytes))
                .expect("Failed to write samples");
            if BALANCED_SAMPLES {
                balanced_samples
            } else {
                samples
            }
        }
    };

    // Make sure all posts are the same average length
    let total_post_length: usize = samples.iter().map(|x| x.lemmas.len()).sum();
    let total_posts: usize = samples.len();
    let avg_post_length = total_post_length / total_posts;
    println!("Average post length: {}", avg_post_length);
    samples.retain(|x| x.lemmas.len() >= avg_post_length);
    for sample in samples.iter_mut() {
        sample.lemmas.truncate(avg_post_length);
    }
    println!("Total posts: {}", samples.len());
    println!(
        "Maximum feature length: {}",
        samples.iter().map(|x| x.lemmas.len()).max().unwrap()
    );
    println!(
        "Minimum feature length: {}",
        samples.iter().map(|x| x.lemmas.len()).min().unwrap()
    );

    // let mut oversampled = oversample(samples);

    println!("Total posts: {}", samples.len());
    println!(
        "Maximum  feature length: {}",
        samples.iter().map(|x| x.lemmas.len()).max().unwrap()
    );
    println!(
        "Minimum feature length: {}",
        samples.iter().map(|x| x.lemmas.len()).min().unwrap()
    );

    // Shuffle the sample set.
    let mut rng = thread_rng();
    samples.shuffle(&mut rng);

    for i in 0..10 {
        println!(
            "{}: {}",
            samples[i].label.to_string(),
            samples[i].lemmas.join(" ")
        );
    }
    samples
}

fn normalize(training_set: &Vec<Sample>) -> (Vec<Vec<f64>>, Vec<u8>) {
    let path_fx = Path::new("./corpus.bincode");
    let path_fy = Path::new("./classifiers.bincode");
    if !path_fx.exists() || !path_fy.exists() {
        println!("Saving x and y matrices...");
        let mut x_set: Vec<Post> = Vec::new();
        for sample in training_set.iter() {
            x_set.push(sample.lemmas.clone());
        }
        let y_set: Vec<u8> = training_set.iter().map(|x| x.label.indicator).collect();
        println!("{} x samples", x_set.len());
        println!("{} y labels", y_set.len());

        let corpus: DMatrix<String> =
            DMatrix::from_fn(x_set.len(), x_set[0].len(), |i, j| x_set[i][j].to_owned());
        let classifiers: DMatrix<u8> = DMatrix::from_fn(y_set.len(), 1, |i, _| y_set[i]);

        // Deterimine unique labels
        let mut unique_labels: Vec<String> = Vec::new();
        for label in y_set.iter() {
            let mbti = MBTI { indicator: *label };
            if !unique_labels.contains(&mbti.to_string()) {
                unique_labels.push(mbti.to_string());
            }
        }
        println!("{} unique labels", unique_labels.len());

        let (dictionary, df_index, personality_freq, overall_freq) = {
            let path_dict: &Path = Path::new("./dictionary.bincode");
            let path_df: &Path = Path::new("./df_index.bincode");
            let path_personality_freq: &Path = Path::new("./personality_freq.bincode");
            let path_overall_freq: &Path = Path::new("./overall_freq.bincode");
            // let path_pos: &Path = Path::new("./pos.bincode");
            // let path_pos_score: &Path = Path::new("./pos_score.bincode");
            if path_dict.exists() && path_df.exists() {
                println!("Loading dictionary...");
                let mut buf = Vec::new();
                File::open(path_dict)
                    .unwrap()
                    .read_to_end(&mut buf)
                    .expect("Unable to read file");
                let dictionary: Dictionary = bincode::deserialize(&buf).unwrap();
                println!("Loading df_index...");
                buf = Vec::new();
                File::open(path_df)
                    .unwrap()
                    .read_to_end(&mut buf)
                    .expect("Unable to read file");
                let df_index: Dictionary = bincode::deserialize(&buf).unwrap();
                println!("Loading personality_freq...");
                buf = Vec::new();
                File::open(path_personality_freq)
                    .unwrap()
                    .read_to_end(&mut buf)
                    .expect("Unable to read file");
                let personality_freq: [Dictionary; 16] = bincode::deserialize(&buf).unwrap();
                println!("Loading overall_freq...");
                buf = Vec::new();
                File::open(path_overall_freq)
                    .unwrap()
                    .read_to_end(&mut buf)
                    .expect("Unable to read file");
                let overall_freq: Dictionary = bincode::deserialize(&buf).unwrap();
                // println!("Loading pos...");
                // buf = Vec::new();
                // File::open(path_pos)
                //     .unwrap()
                //     .read_to_end(&mut buf)
                //     .expect("Unable to read file");
                // let pos: POSDictionary = bincode::deserialize(&buf).unwrap();
                // println!("Loading pos_score...");
                // buf = Vec::new();
                // File::open(path_pos_score)
                //     .unwrap()
                //     .read_to_end(&mut buf)
                //     .expect("Unable to read file");
                // let pos_score: Dictionary = bincode::deserialize(&buf).unwrap();
                (dictionary, df_index, personality_freq, overall_freq)
            } else {
                println!("Saving dictionary and df_index...");
                // Create a dictionary indexing unique tokens.
                // Also create a df_index containing the number of documents each token appears in.
                let mut dictionary: Dictionary = HashMap::new();
                let mut df_major: Dictionary = HashMap::new();
                let mut overall_freq: Dictionary = HashMap::new();
                let mut esfj_freq: Dictionary = HashMap::new();
                let mut esfp_freq: Dictionary = HashMap::new();
                let mut estj_freq: Dictionary = HashMap::new();
                let mut estp_freq: Dictionary = HashMap::new();
                let mut enfj_freq: Dictionary = HashMap::new();
                let mut enfp_freq: Dictionary = HashMap::new();
                let mut entj_freq: Dictionary = HashMap::new();
                let mut entp_freq: Dictionary = HashMap::new();
                let mut isfj_freq: Dictionary = HashMap::new();
                let mut isfp_freq: Dictionary = HashMap::new();
                let mut istj_freq: Dictionary = HashMap::new();
                let mut istp_freq: Dictionary = HashMap::new();
                let mut infj_freq: Dictionary = HashMap::new();
                let mut infp_freq: Dictionary = HashMap::new();
                let mut intj_freq: Dictionary = HashMap::new();
                let mut intp_freq: Dictionary = HashMap::new();
                // let mut pos: POSDictionary = HashMap::new();
                // let mut pos_score: Dictionary = HashMap::new();
                // let pos_model: POSModel = POSModel::new(POSConfig::default()).unwrap();
                for (i, post) in x_set.iter().enumerate() {
                    // let output: Vec<Vec<POSTag>> = pos_model.predict(&post);
                    let df_minor: HashMap<String, f64> =
                        post.iter().fold(HashMap::new(), |mut acc, token| {
                            *acc.entry(token.to_owned()).or_insert(0.0) += 1.0;
                            *overall_freq.entry(token.to_owned()).or_insert(0.0) += 1.0;
                            acc
                        });
                    if y_set[i] == indicator::ESFJ {
                        for (token, count) in df_minor.iter() {
                            *esfj_freq.entry(token.to_owned()).or_insert(0.0) += count;
                        }
                    }
                    if y_set[i] == indicator::ESFP {
                        for (token, count) in df_minor.iter() {
                            *esfp_freq.entry(token.to_owned()).or_insert(0.0) += count;
                        }
                    }
                    if y_set[i] == indicator::ESTJ {
                        for (token, count) in df_minor.iter() {
                            *estj_freq.entry(token.to_owned()).or_insert(0.0) += count;
                        }
                    }
                    if y_set[i] == indicator::ESTP {
                        for (token, count) in df_minor.iter() {
                            *estp_freq.entry(token.to_owned()).or_insert(0.0) += count;
                        }
                    }
                    if y_set[i] == indicator::ENFJ {
                        for (token, count) in df_minor.iter() {
                            *enfj_freq.entry(token.to_owned()).or_insert(0.0) += count;
                        }
                    }
                    if y_set[i] == indicator::ENFP {
                        for (token, count) in df_minor.iter() {
                            *enfp_freq.entry(token.to_owned()).or_insert(0.0) += count;
                        }
                    }
                    if y_set[i] == indicator::ENTJ {
                        for (token, count) in df_minor.iter() {
                            *entj_freq.entry(token.to_owned()).or_insert(0.0) += count;
                        }
                    }
                    if y_set[i] == indicator::ENTP {
                        for (token, count) in df_minor.iter() {
                            *entp_freq.entry(token.to_owned()).or_insert(0.0) += count;
                        }
                    }
                    if y_set[i] == indicator::ISFJ {
                        for (token, count) in df_minor.iter() {
                            *isfj_freq.entry(token.to_owned()).or_insert(0.0) += count;
                        }
                    }
                    if y_set[i] == indicator::ISFP {
                        for (token, count) in df_minor.iter() {
                            *isfp_freq.entry(token.to_owned()).or_insert(0.0) += count;
                        }
                    }
                    if y_set[i] == indicator::ISTJ {
                        for (token, count) in df_minor.iter() {
                            *istj_freq.entry(token.to_owned()).or_insert(0.0) += count;
                        }
                    }
                    if y_set[i] == indicator::ISTP {
                        for (token, count) in df_minor.iter() {
                            *istp_freq.entry(token.to_owned()).or_insert(0.0) += count;
                        }
                    }
                    if y_set[i] == indicator::INFJ {
                        for (token, count) in df_minor.iter() {
                            *infj_freq.entry(token.to_owned()).or_insert(0.0) += count;
                        }
                    }
                    if y_set[i] == indicator::INFP {
                        for (token, count) in df_minor.iter() {
                            *infp_freq.entry(token.to_owned()).or_insert(0.0) += count;
                        }
                    }
                    if y_set[i] == indicator::INTJ {
                        for (token, count) in df_minor.iter() {
                            *intj_freq.entry(token.to_owned()).or_insert(0.0) += count;
                        }
                    }
                    if y_set[i] == indicator::INTP {
                        for (token, count) in df_minor.iter() {
                            *intp_freq.entry(token.to_owned()).or_insert(0.0) += count;
                        }
                    }
                    // assert_eq!(output.len(), post.len());
                    for token in post {
                        if !dictionary.contains_key(&token.to_string()) {
                            dictionary.insert(token.to_string(), dictionary.len() as f64);
                            // if output[j].len() > 0 {
                            // pos.insert(token.to_string(), output[j][0].label.to_string());
                            // pos_score.insert(output[j][0].label.to_string(), pos_score.len() as f64);
                            // }
                        }
                    }
                    df_minor.iter().for_each(|(token, _)| {
                        *df_major.entry(token.to_string()).or_insert(0.0) += 1.0;
                    });
                }
                let personality_freq: [Dictionary; 16] = [
                    esfj_freq, // 0
                    esfp_freq, // 1
                    estj_freq, // 2
                    estp_freq, // 3
                    enfj_freq, // 4
                    enfp_freq, // 5
                    entj_freq, // 6
                    entp_freq, // 7
                    isfj_freq, // 8
                    isfp_freq, // 9
                    istj_freq, // 10
                    istp_freq, // 11
                    infj_freq, // 12
                    infp_freq, // 13
                    intj_freq, // 14
                    intp_freq, // 15
                ];
                // Serialize the dictionary.
                println!("Saving dictionary...");
                let mut f = std::fs::OpenOptions::new()
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .open(path_dict);
                let dictionary_bytes = bincode::serialize(&dictionary).unwrap();
                f.and_then(|mut f| f.write_all(&dictionary_bytes))
                    .expect("Failed to write dictionary");
                // Serialize the df_index.
                println!("Saving df_index...");
                f = std::fs::OpenOptions::new()
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .open(path_df);
                let df_index_bytes = bincode::serialize(&df_major).unwrap();
                f.and_then(|mut f| f.write_all(&df_index_bytes))
                    .expect("Failed to write df_index");
                println!("Saving personality_freq...");
                f = std::fs::OpenOptions::new()
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .open(path_personality_freq);
                let personality_freq_bytes = bincode::serialize(&personality_freq).unwrap();
                f.and_then(|mut f| f.write_all(&personality_freq_bytes))
                    .expect("Failed to write personality_freq");
                println!("Saving overall_freq...");
                f = std::fs::OpenOptions::new()
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .open(path_overall_freq);
                let overall_freq_bytes = bincode::serialize(&overall_freq).unwrap();
                f.and_then(|mut f| f.write_all(&overall_freq_bytes))
                    .expect("Failed to write overall_freq");
                // println!("Saving pos...");
                // f = std::fs::OpenOptions::new()
                //     .write(true)
                //     .create(true)
                //     .truncate(true)
                //     .open(path_pos);
                // let pos_bytes = bincode::serialize(&pos).unwrap();
                // f.and_then(|mut f| f.write_all(&pos_bytes))
                //     .expect("Failed to write pos");
                // println!("Saving pos_score...");
                // f = std::fs::OpenOptions::new()
                //     .write(true)
                //     .create(true)
                //     .truncate(true)
                //     .open(path_pos_score);
                // let pos_score_bytes = bincode::serialize(&pos_score).unwrap();
                // f.and_then(|mut f| f.write_all(&pos_score_bytes))
                //     .expect("Failed to write pos_score");
                (dictionary, df_major, personality_freq, overall_freq)
            }
        };

        println!("Dictionary size: {}", dictionary.len());
        println!("df_index size: {}", df_index.len());
        // println!("pos size: {}", pos.len());

        // Create a tf_idf matrix.

        // Closures to Create TF*IDF matrices from a corpus.
        // tf is the number of times a term appears in a document
        // idf is the inverse document frequency of a term e.g. N divided by how many posts contain the term
        // tf-idf = tf * log(N / df)
        // Where N is number of documents and df is number of documents containing the term.
        let tf = |doc: DMatrixSlice<String>, term: &String| -> f64 {
            DMatrix::from_fn(doc.nrows(), doc.ncols(), |i, j| {
                if doc[(i, j)] == *term {
                    1.0
                } else {
                    0.0
                }
            })
            .sum()
                / doc.ncols() as f64
        };
        let idf = |term: &String| -> f64 {
            // Smooth inverse formula by adding 1.0 to denominator to prevent division by zero
            let score = (corpus.nrows() as f64 / (df_index[term] + 1.0)).ln() as f64;
            score
        };

        // Create a dense matrix of term frequencies.
        let tf_matrix: Vec<Vec<f64>> = {
            let path = Path::new("./tf_matrix.bincode");
            if path.exists() {
                println!("Loading tf_matrix...");
                let mut buf = Vec::new();
                File::open(path)
                    .unwrap()
                    .read_to_end(&mut buf)
                    .expect("Unable to read file");
                let tf_matrix: Vec<Vec<f64>> = bincode::deserialize(&buf).unwrap();
                tf_matrix
            } else {
                println!("Creating a dense matrix of term frequencies...");
                let start = Instant::now();
                let tf_matrix: DMatrix<f64> =
                    DMatrix::from_fn(corpus.nrows(), corpus.ncols(), |i, j| -> f64 {
                        tf(corpus.slice((i, 0), (1, corpus.ncols())), &corpus[(i, j)])
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
                let f = std::fs::OpenOptions::new()
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .open(path);
                let tf_matrix_bytes = bincode::serialize(&corpus_tf).unwrap();
                f.and_then(|mut f| f.write_all(&tf_matrix_bytes))
                    .expect("Failed to write tf_matrix");
                corpus_tf
            }
        };

        // Create a dense matrix of idf values.
        let idf_matrix: Vec<Vec<f64>> = {
            let path = Path::new("./idf_matrix.bincode");
            if path.exists() {
                println!("Loading idf_matrix...");
                let mut buf = Vec::new();
                File::open(path)
                    .unwrap()
                    .read_to_end(&mut buf)
                    .expect("Unable to read file");
                let idf_matrix: Vec<Vec<f64>> = bincode::deserialize(&buf).unwrap();
                idf_matrix
            } else {
                println!("Creating a dense matrix of idf values...");
                let start = Instant::now();
                let idf_matrix: DMatrix<f64> =
                    DMatrix::from_fn(corpus.nrows(), corpus.ncols(), |i, j| -> f64 {
                        idf(&corpus[(i, j)])
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
                let f = std::fs::OpenOptions::new()
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .open(path);
                let idf_matrix_bytes = bincode::serialize(&corpus_idf).unwrap();
                f.and_then(|mut f| f.write_all(&idf_matrix_bytes))
                    .expect("Failed to write idf_matrix");
                corpus_idf
            }
        };

        // Finally, create the tf-idf matrix by multiplying.
        let tf_idf: Vec<Vec<f64>> = {
            let path = Path::new("./tf_idf.bincode");
            if path.exists() {
                println!("Loading tf_idf...");
                let mut buf = Vec::new();
                File::open(path)
                    .unwrap()
                    .read_to_end(&mut buf)
                    .expect("Unable to read file");
                let tf_idf: Vec<Vec<f64>> = bincode::deserialize(&buf).unwrap();
                tf_idf
            } else {
                println!("Creating the tf-idf matrix by multiplying...");
                let start = Instant::now();
                let tf_idf: DMatrix<Features> =
                    DMatrix::from_fn(corpus.nrows(), corpus.ncols(), |i, j| -> Features {
                        let score: f64 = (tf_matrix[i][j] * idf_matrix[i][j]) as f64;
                        score
                    });
                // Normalizing makes the values too tiny.
                let max = tf_idf.max();
                let min = tf_idf.min();
                let _tf_idf_normal: DMatrix<f64> =
                    DMatrix::from_fn(tf_idf.nrows(), tf_idf.ncols(), |i, j| tf_idf[(i, j)] / max);
                println!("tf_idf: {} seconds", start.elapsed().as_secs());
                // Convert tf_idf to a Vec<Vec<f64>>.
                println!("max: {}", max);
                println!("min: {}", min);
                // let df2: DMatrix<f64> = DMatrix::from_fn(tf_idf.nrows(), tf_idf.ncols(), |i, j| tf_idf[(i, j)].1 / dictionary.len() as f64);
                // let df3: DMatrix<f64> = DMatrix::from_fn(tf_idf.nrows(), tf_idf.ncols(), |i, j| tf_idf[(i, j)].2);
                // let grid: [DMatrix<f64>; 3] = [df1, df2, df3];

                // let mut corpus_tf_idf: Vec<Vec<f64>> = Vec::new();
                // for i in 0..df1.nrows() {
                //     let mut row: Vec<f64> = Vec::new();
                //     for y in 0..12 {
                //         for x in 0..12 {
                //             let height: f64 = 12.0 - y as f64;
                //             let freq: f64 = df1[(i, x)];
                //             let size: f64 = 12.0;
                //             let _d_height: Decimal = Decimal::from_f64(height).unwrap();
                //             let _d_freq: Decimal = Decimal::from_f64(tf_idf_normal[(i, x)]).unwrap();
                //             let _id: f64 = df2[(i, x)];
                //             assert!(max < size);
                //             if height >= 0.0 && freq * max > height {
                //                 row.push(1.0);
                //             }
                //             else {
                //                 row.push(0.0);
                //             }
                //         }
                //     }
                //     assert_eq!(row.len(), 144);
                //     corpus_tf_idf.push(row);
                // }

                // Calculate a grid graphed image of the frequency at which each personality type's
                // common terms appear in the document based on the corpus frequency of the word
                // overall.

                // Take the overall score of 12 terms and divide by the maximum score of any 12
                // terms.  Then plot that score's value with precision, 2 decimal places, on a 4x4
                // cell.  The cell with the highest score is the personality type that the document.

                // The graph is a 256 bit grid with 16 cells for each personality type.
                // and so on.

                // The score for each personality type is the sum of the scores of the 12 terms
                // that are most common to that personality type.

                // Calculate the overall frequency as a logarithm of all of the terms used.
                let (overall, overall_terms) = {
                    let path_overall = Path::new("./overall.bincode");
                    let overall: DMatrix<f64>;
                    if !path_overall.exists() {
                        println!("Creating overall...");
                        overall =
                            DMatrix::from_fn(corpus.nrows(), 16, |i, j| {
                                if i + 1 % 25000 == 0 {
                                    println!("overall: {} of {}", i, corpus.nrows());
                                }
                                corpus.row(i).iter().enumerate().fold(
                                    0.0,
                                    |mut acc: f64, (n, term)| {
                                        let mut val: f64 = 0.0;
                                        if personality_freq[j].contains_key(term) {
                                            val = personality_freq[j][term];
                                        }
                                        // TODO We are trying to incorporate term frequency here.
                                        // acc += (val / overall_freq[term]) as f64 * tf_idf[(i, n)];
                                        // df1 is the score where tf-idf is the score and term id in a tuple
                                        acc += ((val / overall_freq[term]) * tf_idf[(i, n)]) as f64;
                                        acc
                                    },
                                ) / corpus.ncols() as f64
                            });
                        let mut obj: Vec<Vec<f64>> = Vec::new();
                        for i in 0..overall.nrows() {
                            let mut row: Vec<f64> = Vec::new();
                            for j in 0..overall.ncols() {
                                row.push(overall[(i, j)]);
                            }
                            obj.push(row);
                        }
                        println!("Saving overall frequencies for each classification...");
                        let overall_bytes = bincode::serialize(&obj).unwrap();
                        let mut f = File::create(path_overall).unwrap();
                        f.write_all(&overall_bytes)
                            .expect("Failed to write overall");
                        println!("Saved overall frequencies for each classification.");
                    } else {
                        println!("Loading overall frequencies for each classification...");
                        let mut buf = Vec::new();
                        File::open(path_overall)
                            .unwrap()
                            .read_to_end(&mut buf)
                            .expect("Unable to read file");
                        let obj: Vec<Vec<f64>> = bincode::deserialize(&buf).unwrap();
                        overall = DMatrix::from_fn(obj.len(), obj[0].len(), |i, j| obj[i][j]);
                    }
                    let overall_terms = DMatrix::from_fn(corpus.nrows(), corpus.ncols(), |i, j| {
                        if i + 1 % 25000 == 0 {
                            println!("overall_terms: {} of {}", i, corpus.nrows());
                        }
                        let term = &corpus[(i, j)];
                        // a tuple containing 16 values, one for each personality type.
                        let mut val: [f64; 16] = [0.0; 16];
                        for x in 0..16 {
                            const MODE_TERM_FREQ_VS_OVERALL_FREQ: bool = false;
                            if MODE_TERM_FREQ_VS_OVERALL_FREQ {
                                if personality_freq[x].contains_key(term) {
                                    val[x] = (personality_freq[x][term] / overall_freq[term])
                                        * tf_idf[(i, j)] as f64; // Don't do tf_idf_normal here, because it makes the value to small.
                                } else {
                                    val[x] = 0.0;
                                }
                            } else {
                                if personality_freq[x].contains_key(term) {
                                    val[x] =
                                        (personality_freq[x][term] / overall_freq[term]) as f64;
                                } else {
                                    val[x] = 0.0;
                                }
                            }
                        }
                        val
                    });
                    (overall, overall_terms)
                };

                // for i in 0..corpus.nrows() {
                //     for j in 0..corpus.ncols() {
                //         let term = &corpus[(i, j)];
                //         for n in 0..16 {
                //             let mut val = 0.0;
                //             if personality_freq[n].contains_key(term) {
                //                 val = personality_freq[n][term];
                //             }
                //             let score: f64 = val / overall_freq[term] as f64;
                //             overall[(i,n)] += score;
                //         }
                //     }
                // }
                println!("Starting heatmap");
                let mut max_height = 0;
                let mut max = [0.0; 8];
                let mut max_avg = [0.0; 8];
                let mut max_sum = [0.0; 8];
                let heatmap: Vec<Vec<f64>> = {
                    let path_heatmap = Path::new("./heatmap.bincode");
                    let path_classifiers1 = Path::new("./classifiers1.bincode");
                    let mut heatmap: Vec<Vec<f64>>;
                    let mut classifiers1: Vec<u8> = Vec::new();
                    if !path_heatmap.exists() {
                        // Calculate the 256 bit grid.
                        heatmap = Vec::new();
                        let max_overall = overall.max();
                        for i in 0..corpus.nrows() {
                            let mut maxes: [f64; 4] = [0.0; 4];
                            let _image16: DMatrix<f64> = DMatrix::from_fn(16, 16, |y, x| {
                                let n: usize;
                                // The first personality type, ESFJ, is the first 16 bits.
                                // [(0,0), (0,1), (0,2), (0,3),
                                //  (1,0), (1,1), (1,2), (1,3),
                                //  (2,0), (2,1), (2,2), (2,3),
                                //  (3,0), (3,1), (3,2), (3,3)]
                                if x <= 3 && y <= 3 {
                                    n = 0;
                                }
                                // The second personality type, ESFP, is the second 16 bits.
                                // [(4,0), (4,1), (4,2), (4,3),
                                //  (5,0), (5,1), (5,2), (5,3),
                                //  (6,0), (6,1), (6,2), (6,3),
                                //  (7,0), (7,1), (7,2), (7,3)]
                                else if x >= 4 && x <= 7 && y <= 3 {
                                    n = 1;
                                }
                                // The third personality type, ESTJ, is the third 16 bits.
                                // [(8,0), (8,1), (8,2), (8,3),
                                //  (9,0), (9,1), (9,2), (9,3),
                                //  (10,0), (10,1), (10,2), (10,3),
                                //  (11,0), (11,1), (11,2), (11,3)]
                                else if x >= 8 && x <= 11 && y <= 3 {
                                    n = 2;
                                }
                                // The fourth personality type, ESTP, is the fourth 16 bits.
                                // [(12,0), (12,1), (12,2), (12,3),
                                //  (13,0), (13,1), (13,2), (13,3),
                                //  (14,0), (14,1), (14,2), (14,3),
                                //  (15,0), (15,1), (15,2), (15,3)]
                                else if x >= 12 && x <= 15 && y <= 3 {
                                    n = 3;
                                }
                                // The fifth personality type, ENFJ, is the fifth 16 bits.
                                // [(0,4), (0,5), (0,6), (0,7),
                                //  (1,4), (1,5), (1,6), (1,7),
                                //  (2,4), (2,5), (2,6), (2,7),
                                //  (3,4), (3,5), (3,6), (3,7)]
                                else if x <= 3 && y >= 4 && y <= 7 {
                                    n = 4;
                                }
                                // The sixth personality type, ENFP, is the sixth 16 bits.:
                                // [(4,4), (4,5), (4,6), (4,7),
                                //  (5,4), (5,5), (5,6), (5,7),
                                //  (6,4), (6,5), (6,6), (6,7),
                                //  (7,4), (7,5), (7,6), (7,7)]
                                else if x >= 4 && x <= 7 && y >= 4 && y <= 7 {
                                    n = 5;
                                }
                                // The seventh personality type, ENTJ, is the seventh 16 bits.
                                // [(8,4), (8,5), (8,6), (8,7),
                                //  (9,4), (9,5), (9,6), (9,7),
                                //  (10,4), (10,5), (10,6), (10,7),
                                //  (11,4), (11,5), (11,6), (11,7)]
                                else if x >= 8 && x <= 11 && y >= 4 && y <= 7 {
                                    n = 6;
                                }
                                // The eighth personality type, ENTP, is the eighth 16 bits.
                                // [(12,4), (12,5), (12,6), (12,7),
                                //  (13,4), (13,5), (13,6), (13,7),
                                //  (14,4), (14,5), (14,6), (14,7),
                                //  (15,4), (15,5), (15,6), (15,7)]
                                else if x >= 12 && x <= 15 && y >= 4 && y <= 7 {
                                    n = 7;
                                }
                                // The ninth personality type, ISFJ, is the ninth 16 bits.:
                                // [(0,8), (0,9), (0,10), (0,11),
                                //  (1,8), (1,9), (1,10), (1,11),
                                //  (2,8), (2,9), (2,10), (2,11),
                                //  (3,8), (3,9), (3,10), (3,11)]
                                else if x <= 3 && y >= 8 && y <= 11 {
                                    n = 8;
                                }
                                // The tenth personality type, ISFP, is the tenth 16 bits.:
                                // [(4,8), (4,9), (4,10), (4,11),
                                //  (5,8), (5,9), (5,10), (5,11),
                                //  (6,8), (6,9), (6,10), (6,11),
                                //  (7,8), (7,9), (7,10), (7,11)]
                                else if x >= 4 && x <= 7 && y >= 8 && y <= 11 {
                                    n = 9;
                                }
                                // The eleventh personality type, ISTJ, is the eleventh 16 bits.
                                // [(8,8), (8,9), (8,10), (8,11),
                                //  (9,8), (9,9), (9,10), (9,11),
                                //  (10,8), (10,9), (10,10), (10,11),
                                //  (11,8), (11,9), (11,10), (11,11)]
                                else if x >= 8 && x <= 11 && y >= 8 && y <= 11 {
                                    n = 10;
                                }
                                // The twelfth personality type, ISTP, is the twelfth 16 bits.
                                // [(12,8), (12,9), (12,10), (12,11),
                                //  (13,8), (13,9), (13,10), (13,11),
                                //  (14,8), (14,9), (14,10), (14,11),
                                //  (15,8), (15,9), (15,10), (15,11)]
                                else if x >= 12 && x <= 15 && y >= 8 && y <= 11 {
                                    n = 11;
                                }
                                // The thirteenth personality type, INFJ, is the thirteenth 16 bits.
                                // [(0,12), (0,13), (0,14), (0,15),
                                //  (1,12), (1,13), (1,14), (1,15),
                                //  (2,12), (2,13), (2,14), (2,15),
                                //  (3,12), (3,13), (3,14), (3,15)]
                                else if x <= 3 && y >= 12 && y <= 15 {
                                    n = 12;
                                }
                                // The fourteenth personality type, INFP, is the fourteenth 16 bits.
                                // [(4,12), (4,13), (4,14), (4,15),
                                //  (5,12), (5,13), (5,14), (5,15),
                                //  (6,12), (6,13), (6,14), (6,15),
                                //  (7,12), (7,13), (7,14), (7,15)]
                                else if x >= 4 && x <= 7 && y >= 12 && y <= 15 {
                                    n = 13;
                                }
                                // The fifteenth personality type, INTJ, is the fifteenth 16 bits.
                                // [(8,12), (8,13), (8,14), (8,15),
                                //  (9,12), (9,13), (9,14), (9,15),
                                //  (10,12), (10,13), (10,14), (10,15),
                                //  (11,12), (11,13), (11,14), (11,15)]
                                else if x >= 8 && x <= 11 && y >= 12 && y <= 15 {
                                    n = 14;
                                }
                                // The sixteenth personality type, INTP, is the sixteenth 16 bits.
                                // [(12,12), (12,13), (12,14), (12,15),
                                //  (13,12), (13,13), (13,14), (13,15),
                                //  (14,12), (14,13), (14,14), (14,15),
                                //  (15,12), (15,13), (15,14), (15,15)]
                                else if x >= 12 && x <= 15 && y >= 12 && y <= 15 {
                                    n = 15;
                                } else {
                                    println!("x: {}, y: {}", x, y);
                                    panic!("Invalid x and y values.");
                                }
                                overall[(i, n)] / max_overall
                            });
                            let image8: DMatrix<f64> = DMatrix::from_fn(16, 16, |y, x| {
                                const DIRECT_VS_AVG: bool = true;
                                let index: [usize; 8];
                                let n: usize;
                                let shape: [usize; 2];
                                let mut avg: f64 = 0.0;
                                let mut score: f64 = 0.0;
                                // Alternate x and y values in each quadrant to improve readability.
                                // The first 64 bits rectangle (8x8) is the indicator IE.
                                // [(0,0), (0,1), (0,2), (0,3), (0,4), (0,5), (0,6), (0,7),
                                //  (1,0), (1,1), (1,2), (1,3), (1,4), (1,5), (1,6), (1,7),
                                //  (2,0), (2,1), (2,2), (2,3), (2,4), (2,5), (2,6), (2,7),
                                //  (3,0), (3,1), (3,2), (3,3), (3,4), (3,5), (3,6), (3,7),
                                //  (4,0), (4,1), (4,2), (4,3), (4,4), (4,5), (4,6), (4,7),
                                //  (5,0), (5,1), (5,2), (5,3), (5,4), (5,5), (5,6), (5,7),
                                //  (6,0), (6,1), (6,2), (6,3), (6,4), (6,5), (6,6), (6,7),
                                //  (7,0), (7,1), (7,2), (7,3), (7,4), (7,5), (7,6), (7,7)]
                                // First do [(0,0), (7,3)].
                                // The first 32 bits rectangle (8x4) is the indicator I.
                                if x <= 3 && y <= 7 {
                                    index = [8, 9, 10, 11, 12, 13, 14, 15];
                                    n = 0;
                                    shape = [8, 4];
                                }
                                // Then do [(0,4), (7,7)].
                                // The second 32 bits rectangle (8x4) is the indicator E.
                                else if x >= 4 && x <= 7 && y <= 7 {
                                    index = [0, 1, 2, 3, 4, 5, 6, 7];
                                    n = 0;
                                    shape = [8, 4];
                                }
                                // The second 64 bits rectangle (8x8) is the indicator SN.
                                // [(0,8), (0,9), (0,10), (0,11), (0,12), (0,13), (0,14), (0,15),
                                //  (1,8), (1,9), (1,10), (1,11), (1,12), (1,13), (1,14), (1,15),
                                //  (2,8), (2,9), (2,10), (2,11), (2,12), (2,13), (2,14), (2,15),
                                //  (3,8), (3,9), (3,10), (3,11), (3,12), (3,13), (3,14), (3,15),
                                //  (4,8), (4,9), (4,10), (4,11), (4,12), (4,13), (4,14), (4,15),
                                //  (5,8), (5,9), (5,10), (5,11), (5,12), (5,13), (5,14), (5,15),
                                //  (6,8), (6,9), (6,10), (6,11), (6,12), (6,13), (6,14), (6,15),
                                //  (7,8), (7,9), (7,10), (7,11), (7,12), (7,13), (7,14), (7,15)]
                                // First do [(0,8), (3,15)].
                                // The third 32 bits rectangle (4x8) is the indicator S.
                                else if x >= 8 && x <= 15 && y <= 3 {
                                    index = [0, 1, 2, 3, 8, 9, 10, 11];
                                    n = 1;
                                    shape = [4, 8];
                                }
                                // Then do [(4,8), (7,15)].
                                // The fourth 32 bits rectangle (4x8) is the indicator N.
                                else if x >= 8 && x <= 15 && y >= 4 && y <= 7 {
                                    index = [4, 5, 6, 7, 12, 13, 14, 15];
                                    n = 1;
                                    shape = [4, 8];
                                }
                                // The third 64 bits rectangle (8x8) is the indicator TF.
                                // [(8,0), (8,1), (8,2), (8,3), (8,4), (8,5), (8,6), (8,7),
                                //  (9,0), (9,1), (9,2), (9,3), (9,4), (9,5), (9,6), (9,7),
                                //  (10,0), (10,1), (10,2), (10,3), (10,4), (10,5), (10,6), (10,7),
                                //  (11,0), (11,1), (11,2), (11,3), (11,4), (11,5), (11,6), (11,7),
                                //  (12,0), (12,1), (12,2), (12,3), (12,4), (12,5), (12,6), (12,7),
                                //  (13,0), (13,1), (13,2), (13,3), (13,4), (13,5), (13,6), (13,7),
                                //  (14,0), (14,1), (14,2), (14,3), (14,4), (14,5), (14,6), (14,7),
                                //  (15,0), (15,1), (15,2), (15,3), (15,4), (15,5), (15,6), (15,7)]
                                // First do [(8,0), (11,7)].
                                // The fifth 32 bits rectangle (8x4) is the indicator T.
                                else if x <= 7 && y >= 8 && y <= 11 {
                                    index = [2, 3, 6, 7, 10, 11, 14, 15];
                                    n = 2;
                                    shape = [4, 8];
                                }
                                // Then do [(12,0), (15,7)].
                                // The sixth 32 bits rectangle (8x4) is the indicator F.
                                else if x <= 7 && y >= 12 && y <= 15 {
                                    index = [0, 1, 4, 5, 8, 9, 12, 13];
                                    n = 2;
                                    shape = [4, 8];
                                }
                                // The fourth 64 bits square (8x8) is the indicator JP.
                                // [(8,8), (8,9), (8,10), (8,11), (8,12), (8,13), (8,14), (8,15),
                                //  (9,8), (9,9), (9,10), (9,11), (9,12), (9,13), (9,14), (9,15),
                                //  (10,8), (10,9), (10,10), (10,11), (10,12), (10,13), (10,14), (10,15),
                                //  (11,8), (11,9), (11,10), (11,11), (11,12), (11,13), (11,14), (11,15),
                                //  (12,8), (12,9), (12,10), (12,11), (12,12), (12,13), (12,14), (12,15),
                                //  (13,8), (13,9), (13,10), (13,11), (13,12), (13,13), (13,14), (13,15),
                                //  (14,8), (14,9), (14,10), (14,11), (14,12), (14,13), (14,14), (14,15),
                                //  (15,8), (15,9), (15,10), (15,11), (15,12), (15,13), (15,14), (15,15)]
                                // First do [(8,8), (15,11)].
                                // The seventh 32 bits rectangle (4x8) is the indicator J.
                                else if x >= 8 && x <= 11 && y >= 8 && y <= 15 {
                                    index = [0, 2, 4, 6, 8, 10, 12, 14];
                                    n = 3;
                                    shape = [8, 4];
                                }
                                // Then do [(8, 12), (15,15)].
                                // The eighth 32 bits rectangle (4x8) is the indicator P.
                                else if x >= 12 && x <= 15 && y >= 8 && y <= 15 {
                                    index = [1, 3, 5, 7, 9, 11, 13, 15];
                                    n = 3;
                                    shape = [8, 4];
                                } else {
                                    panic!("Invalid coordinates: ({}, {})", x, y);
                                }
                                {
                                    if DIRECT_VS_AVG {
                                        for delta in index.iter() {
                                            // translate x and y into coordinates for overall_terms's shape.
                                            // find the relative position in the overall_term subvector with cols corpus.ncols()
                                            // and rows corpus.nrows().
                                            let relative_y = y % shape[0];
                                            let relative_x = x % shape[1];
                                            score += overall_terms
                                                [(i, (relative_y + 1) * (relative_x + 1) - 1)]
                                                [*delta];
                                        }
                                        if score > maxes[n] {
                                            maxes[n] = score;
                                        }
                                        score
                                    } else {
                                        for delta in index.iter() {
                                            avg += overall[(i, *delta)] / max_overall;
                                        }
                                        avg /= index.len() as f64;
                                        if avg > maxes[n] {
                                            maxes[n] = avg;
                                        }
                                        avg
                                    }
                                }
                            });

                            let soft_gradient = |x: f64, max: f64| -> f64 {
                                if x == max {
                                    1.0
                                } else if x / max >= 0.77 {
                                    0.77
                                } else if x / max >= 0.33 {
                                    0.33
                                } else {
                                    0.0
                                }
                            };

                            let _image8_normalized =
                                DMatrix::from_fn(image8.nrows(), image8.ncols(), |y, x| {
                                    let norm_i = image8.slice((0, 0), (8, 4)).normalize();
                                    let norm_e = image8.slice((0, 4), (8, 4)).normalize();
                                    let norm_s = image8.slice((0, 8), (4, 8)).normalize();
                                    let norm_n = image8.slice((4, 8), (4, 8)).normalize();
                                    let norm_t = image8.slice((8, 0), (4, 8)).normalize();
                                    let norm_f = image8.slice((12, 0), (4, 8)).normalize();
                                    let norm_j = image8.slice((8, 8), (8, 4)).normalize();
                                    let norm_p = image8.slice((8, 12), (8, 4)).normalize();
                                    let avg_i = norm_i.mean();
                                    let avg_e = norm_e.mean();
                                    let avg_s = norm_s.mean();
                                    let avg_n = norm_n.mean();
                                    let avg_t = norm_t.mean();
                                    let avg_f = norm_f.mean();
                                    let avg_j = norm_j.mean();
                                    let avg_p = norm_p.mean();
                                    let sum_i = norm_i.sum();
                                    let sum_e = norm_e.sum();
                                    let sum_s = norm_s.sum();
                                    let sum_n = norm_n.sum();
                                    let sum_t = norm_t.sum();
                                    let sum_f = norm_f.sum();
                                    let sum_j = norm_j.sum();
                                    let sum_p = norm_p.sum();
                                    let max_i = norm_i.max();
                                    let max_e = norm_e.max();
                                    let max_s = norm_s.max();
                                    let max_n = norm_n.max();
                                    let max_t = norm_t.max();
                                    let max_f = norm_f.max();
                                    let max_j = norm_j.max();
                                    let max_p = norm_p.max();

                                    if max_i > max[0] {
                                        max[0] = max_i;
                                    }
                                    if max_e > max[1] {
                                        max[1] = max_e;
                                    }
                                    if max_s > max[2] {
                                        max[2] = max_s;
                                    }
                                    if max_n > max[3] {
                                        max[3] = max_n;
                                    }
                                    if max_t > max[4] {
                                        max[4] = max_t;
                                    }
                                    if max_f > max[5] {
                                        max[5] = max_f;
                                    }
                                    if max_j > max[6] {
                                        max[6] = max_j;
                                    }
                                    if max_p > max[7] {
                                        max[7] = max_p;
                                    }
                                    if avg_i > max_avg[0] {
                                        max_avg[0] = avg_i;
                                    }
                                    if avg_e > max_avg[1] {
                                        max_avg[1] = avg_e;
                                    }
                                    if avg_s > max_avg[2] {
                                        max_avg[2] = avg_s;
                                    }
                                    if avg_n > max_avg[3] {
                                        max_avg[3] = avg_n;
                                    }
                                    if avg_t > max_avg[4] {
                                        max_avg[4] = avg_t;
                                    }
                                    if avg_f > max_avg[5] {
                                        max_avg[5] = avg_f;
                                    }
                                    if avg_j > max_avg[6] {
                                        max_avg[6] = avg_j;
                                    }
                                    if avg_p > max_avg[7] {
                                        max_avg[7] = avg_p;
                                    }
                                    if sum_i > max_sum[0] {
                                        max_sum[0] = sum_i;
                                    }
                                    if sum_e > max_sum[1] {
                                        max_sum[1] = sum_e;
                                    }
                                    if sum_s > max_sum[2] {
                                        max_sum[2] = sum_s;
                                    }
                                    if sum_n > max_sum[3] {
                                        max_sum[3] = sum_n;
                                    }
                                    if sum_t > max_sum[4] {
                                        max_sum[4] = sum_t;
                                    }
                                    if sum_f > max_sum[5] {
                                        max_sum[5] = sum_f;
                                    }
                                    if sum_j > max_sum[6] {
                                        max_sum[6] = sum_j;
                                    }
                                    if sum_p > max_sum[7] {
                                        max_sum[7] = sum_p;
                                    }

                                    if x < 4 && y < 8 {
                                        norm_i[(y, x)]
                                    } else if x >= 4 && x < 8 && y < 8 {
                                        norm_e[(y, x - 4)]
                                    } else if x >= 8 && x < 16 && y < 4 {
                                        norm_s[(y, x - 8)]
                                    } else if x >= 8 && x < 16 && y < 8 {
                                        norm_n[(y - 4, x - 8)]
                                    } else if x < 8 && y >= 8 && y < 12 {
                                        norm_t[(y - 8, x)]
                                    } else if x < 8 && y >= 12 {
                                        norm_f[(y - 12, x)]
                                    } else if x >= 8 && x < 12 && y >= 8 {
                                        norm_j[(y - 8, x - 8)]
                                    } else if x >= 12 && x < 16 && y >= 8 {
                                        norm_p[(y - 8, x - 12)]
                                    } else {
                                        panic!("Invalid coordinates: ({}, {})", x, y);
                                    }
                                });

                            let segment_normalized1: DMatrix<f64> =
                                DMatrix::from_fn(image8.nrows(), image8.ncols(), |y, x| {
                                    const SOFTGRADIENT: bool = false;
                                    const HOTENCODED: bool = true;
                                    const SINEWAVE: bool = false;
                                    const WAVE: bool = false;
                                    const RADIUS: bool = false;
                                    let mut val: f64 = 0.0;
                                    let norm_i =
                                        image8.normalize().slice((0, 0), (8, 4)).normalize();
                                    let norm_e =
                                        image8.normalize().slice((0, 4), (8, 4)).normalize();
                                    let norm_s =
                                        image8.normalize().slice((0, 8), (4, 8)).normalize();
                                    let norm_n =
                                        image8.normalize().slice((4, 8), (4, 8)).normalize();
                                    let norm_t =
                                        image8.normalize().slice((8, 0), (4, 8)).normalize();
                                    let norm_f =
                                        image8.normalize().slice((12, 0), (4, 8)).normalize();
                                    let norm_j =
                                        image8.normalize().slice((8, 8), (8, 4)).normalize();
                                    let norm_p =
                                        image8.normalize().slice((8, 12), (8, 4)).normalize();
                                    let _avg_i = norm_i.mean();
                                    let _avg_e = norm_e.mean();
                                    let _avg_s = norm_s.mean();
                                    let _avg_n = norm_n.mean();
                                    let _avg_t = norm_t.mean();
                                    let _avg_f = norm_f.mean();
                                    let _avg_j = norm_j.mean();
                                    let _avg_p = norm_p.mean();
                                    let sum_i = norm_i.sum();
                                    let sum_e = norm_e.sum();
                                    let sum_s = norm_s.sum();
                                    let sum_n = norm_n.sum();
                                    let sum_t = norm_t.sum();
                                    let sum_f = norm_f.sum();
                                    let sum_j = norm_j.sum();
                                    let sum_p = norm_p.sum();
                                    let _max_i = norm_i.max();
                                    let _max_e = norm_e.max();
                                    let _max_s = norm_s.max();
                                    let _max_n = norm_n.max();
                                    let _max_t = norm_t.max();
                                    let _max_f = norm_f.max();
                                    let _max_j = norm_j.max();
                                    let _max_p = norm_p.max();

                                    // let _max =
                                    //     [avg_i, avg_e, avg_s, avg_n, avg_t, avg_f, avg_j, avg_p]
                                    //         .iter()
                                    //         .fold(0.0, |acc: f64, x: &f64| acc.max(*x));

                                    // Calculate the radius at a given quadrant.  The quadrant will be split at a 45 degree
                                    // angle from the center point of either (7, 7), (7, 8), (8, 7), (8, 8) for (y, x).
                                    // The radius is the distance from the center point to the edge of the quadrant.
                                    // a true upper_45 value when x <= 7 and y <= 7
                                    // means the value falls above the 45 degree slope from (7,7) towards (0, 0).
                                    // A false upper_45 value when x <= 7 and y <= 7
                                    // means the value falls below the 45 degree slope from (7,7) towards (0, 0).
                                    // A true upper_45 value when x > 7 and y <= 7
                                    // means the value falls above the 45 degree slope from (7,8) towards (0, 15).
                                    // A false upper_45 value when x > 7 and y <= 7
                                    // means the value falls below the 45 degree slope from (8,7) towards (0, 15).
                                    // A true upper_45 value when x <= 7 and y > 7
                                    // means the value falls above the 45 degree slope from (8,7) towards (15, 0).
                                    // A false upper_45 value when x <= 7 and y > 7
                                    // means the value falls below the 45 degree slope from (8,7) towards (15, 0).
                                    // A true upper_45 value when x > 7 and y > 7
                                    // means the value falls above the 45 degree slope from (8,8) towards (15, 15).
                                    // A false upper_45 value when x > 7 and y > 7
                                    // means the value falls below the 45 degree slope from (8,8) towards (15, 15).
                                    let radial_distance_xy = |x: usize, y: usize, angle: usize| {
                                        let yy = y as f64;
                                        let xx = x as f64;
                                        let mut distance: f64 = 0.0;
                                        if angle == 315 {
                                            let x1 = 7.0;
                                            let y1 = 7.0;
                                            distance =
                                                ((yy - y1).powi(2) + (xx - x1).powi(2)).sqrt();
                                        } else if angle == 45 {
                                            let x1 = 8.0;
                                            let y1 = 7.0;
                                            distance =
                                                ((yy - y1).powi(2) + (xx - x1).powi(2)).sqrt();
                                        } else if angle == 135 {
                                            let x1 = 7.0;
                                            let y1 = 8.0;
                                            distance =
                                                ((yy - y1).powi(2) + (xx - x1).powi(2)).sqrt();
                                        } else if angle == 225 {
                                            let x1 = 8.0;
                                            let y1 = 8.0;
                                            distance =
                                                ((yy - y1).powi(2) + (xx - x1).powi(2)).sqrt();
                                        }
                                        distance
                                    };
                                    let height_at_x =
                                        |x, start, end: usize, x_start, x_end| -> (usize, f64) {
                                            let initial_height = start as f64;
                                            let final_height = end as f64;
                                            let initial_x = x_start as f64;
                                            let final_x = x_end as f64;
                                            let slope = if WAVE {
                                                (final_height - initial_height)
                                                    / (final_x - initial_x)
                                            } else {
                                                0.0
                                            };
                                            let y = slope * (x as f64 - initial_x) + initial_height;
                                            let new_y: usize = if y.round() > 15.0 {
                                                15
                                            } else {
                                                y.round() as usize
                                            };
                                            (new_y, slope)
                                        };
                                    let mut ie_height = if sum_i > sum_e {
                                        8.0 + ((1.0 - ((sum_i / sum_i) * (sum_e / sum_i))) * 16.0)
                                    } else {
                                        7.0 - ((1.0 - ((sum_i / sum_e) * (sum_e / sum_e))) * 16.0)
                                    }
                                        as usize;
                                    let mut sn_height = if sum_s > sum_n {
                                        8.0 + ((1.0 - ((sum_s / sum_s) * (sum_n / sum_s))) * 16.0)
                                    } else {
                                        7.0 - ((1.0 - ((sum_s / sum_n) * (sum_n / sum_n))) * 16.0)
                                    }
                                        as usize;
                                    let mut tf_height = if sum_t > sum_f {
                                        8.0 + ((1.0 - ((sum_t / sum_t) * (sum_f / sum_t))) * 16.0)
                                    } else {
                                        7.0 - ((1.0 - ((sum_t / sum_f) * (sum_f / sum_f))) * 16.0)
                                    }
                                        as usize;
                                    let mut jp_height = if sum_j > sum_p {
                                        8.0 + ((1.0 - ((sum_j / sum_j) * (sum_p / sum_j))) * 16.0)
                                    } else {
                                        7.0 - ((1.0 - ((sum_j / sum_p) * (sum_p / sum_p))) * 16.0)
                                    }
                                        as usize;

                                    if ie_height > 15 {
                                        ie_height = 15;
                                    } else if sn_height > 15 {
                                        sn_height = 15;
                                    } else if tf_height > 15 {
                                        tf_height = 15;
                                    } else if jp_height > 15 {
                                        jp_height = 15;
                                    }
                                    ie_height = ie_height / 16 * 7;
                                    sn_height = sn_height / 16 * 8;
                                    tf_height = tf_height / 16 * 8;
                                    jp_height = jp_height / 16 * 7;

                                    if ie_height > max_height {
                                        max_height = ie_height;
                                    }
                                    if sn_height > max_height {
                                        max_height = sn_height;
                                    }
                                    if tf_height > max_height {
                                        max_height = tf_height;
                                    }
                                    if jp_height > max_height {
                                        max_height = jp_height;
                                    }

                                    let ie_start = 0;
                                    let ie_end = ie_start + 3;
                                    let sn_start = ie_start + 4;
                                    let sn_end = sn_start + 3;
                                    let tf_start = sn_start + 4;
                                    let tf_end = tf_start + 3;
                                    let jp_start = tf_start + 4;
                                    let jp_end = jp_start + 3;
                                    let (
                                        mut i_height_y,
                                        mut e_height_y,
                                        mut s_height_y,
                                        mut n_height_y,
                                        mut t_height_y,
                                        mut f_height_y,
                                        mut j_height_y,
                                        mut p_height_y,
                                    ) = if WAVE {
                                        let (ie_height_y, _ie_slope) = height_at_x(
                                            x, ie_height, sn_height, ie_start, sn_start,
                                        );
                                        let (sn_height_y, _sn_slope) = height_at_x(
                                            x, sn_height, tf_height, sn_start, tf_start,
                                        );
                                        let (tf_height_y, _tf_slope) = height_at_x(
                                            x, tf_height, jp_height, tf_start, jp_start,
                                        );
                                        let (jp_height_y, _jp_slope) =
                                            height_at_x(x, jp_height, jp_height, jp_start, jp_end);
                                        (
                                            (ie_height_y, 0),
                                            (0, 0),
                                            (sn_height_y, 0),
                                            (0, 0),
                                            (tf_height_y, 0),
                                            (0, 0),
                                            (jp_height_y, 0),
                                            (0, 0),
                                        )
                                    } else if RADIUS {
                                        let i_height_y =
                                            radial_distance_xy(x, y, 315).round() as usize;
                                        let e_height_y =
                                            radial_distance_xy(x, y, 315).round() as usize;
                                        let s_height_y =
                                            radial_distance_xy(x, y, 45).round() as usize;
                                        let n_height_y =
                                            radial_distance_xy(x, y, 45).round() as usize;
                                        let t_height_y =
                                            radial_distance_xy(x, y, 135).round() as usize;
                                        let f_height_y =
                                            radial_distance_xy(x, y, 135).round() as usize;
                                        let j_height_y =
                                            radial_distance_xy(x, y, 225).round() as usize;
                                        let p_height_y =
                                            radial_distance_xy(x, y, 225).round() as usize;
                                        let i_y = if sum_i > sum_e { 7.0 } else { 0.0 };
                                        let e_y = if sum_i < sum_e { 7.0 } else { 0.0 };
                                        let s_y = if sum_s > sum_n { 8.0 } else { 0.0 };
                                        let n_y = if sum_s < sum_n { 8.0 } else { 0.0 };
                                        let t_y = if sum_t > sum_f { 8.0 } else { 0.0 };
                                        let f_y = if sum_t < sum_f { 8.0 } else { 0.0 };
                                        let j_y = if sum_j > sum_p { 7.0 } else { 0.0 };
                                        let p_y = if sum_j < sum_p { 7.0 } else { 0.0 };
                                        (
                                            (i_height_y, i_y as usize),
                                            (e_height_y, e_y as usize),
                                            (s_height_y, s_y as usize),
                                            (n_height_y, n_y as usize),
                                            (t_height_y, t_y as usize),
                                            (f_height_y, f_y as usize),
                                            (j_height_y, j_y as usize),
                                            (p_height_y, p_y as usize),
                                        )
                                    } else {
                                        let (ie_height_y, _ie_slope) =
                                            height_at_x(x, ie_height, ie_height, ie_start, ie_end);
                                        let (sn_height_y, _sn_slope) =
                                            height_at_x(x, sn_height, sn_height, sn_start, sn_end);
                                        let (tf_height_y, _tf_slope) =
                                            height_at_x(x, tf_height, tf_height, tf_start, tf_end);
                                        let (jp_height_y, _jp_slope) =
                                            height_at_x(x, jp_height, jp_height, jp_start, jp_end);
                                        (
                                            (ie_height_y, 0),
                                            (0, 0),
                                            (sn_height_y, 0),
                                            (0, 0),
                                            (tf_height_y, 0),
                                            (0, 0),
                                            (jp_height_y, 0),
                                            (0, 0),
                                        )
                                    };

                                    if i_height_y.0 > 15 {
                                        i_height_y.0 = 15;
                                    }

                                    if s_height_y.0 > 15 {
                                        s_height_y.0 = 15;
                                    }

                                    if t_height_y.0 > 15 {
                                        t_height_y.0 = 15;
                                    }

                                    if j_height_y.0 > 15 {
                                        j_height_y.0 = 15;
                                    }

                                    if RADIUS {
                                        if e_height_y.0 > 15 {
                                            e_height_y.0 = 15;
                                        }
                                        if n_height_y.0 > 15 {
                                            n_height_y.0 = 15;
                                        }
                                        if f_height_y.0 > 15 {
                                            f_height_y.0 = 15;
                                        }
                                        if p_height_y.0 > 15 {
                                            p_height_y.0 = 15;
                                        }
                                    }

                                    if SINEWAVE {
                                        let _y_float = y as f64;
                                        if x >= ie_start && x <= ie_end {
                                            assert!(i_height_y.0 >= 15);
                                            let _ie_height_float = i_height_y.0 as f64;
                                            if (y <= 7 && i_height_y.0 >= 7 && y >= i_height_y.0)
                                                || (y >= 8
                                                    && i_height_y.0 >= 8
                                                    && y <= i_height_y.0)
                                            {
                                                val = 1.0;
                                            } else {
                                                val = 0.0
                                            }
                                        }
                                        if x >= sn_start && x <= sn_end {
                                            assert!(s_height_y.0 >= 15);
                                            let _sn_height_float = s_height_y.0 as f64;
                                            if (y <= 7 && s_height_y.0 >= 7 && y >= s_height_y.0)
                                                || (y >= 8
                                                    && s_height_y.0 >= 8
                                                    && y <= s_height_y.0)
                                            {
                                                val = 1.0;
                                            } else {
                                                val = 0.0
                                            }
                                        }
                                        if x >= tf_start && x <= tf_end {
                                            assert!(t_height_y.0 >= 15);
                                            let _tf_height_float = t_height_y.0 as f64;
                                            if y <= 7 && t_height_y.0 >= 7 && y >= t_height_y.0
                                                || y >= 8 && t_height_y.0 >= 8 && y <= t_height_y.0
                                            {
                                                val = 1.0;
                                            } else {
                                                val = 0.0
                                            }
                                        }
                                        if x >= jp_start && x <= jp_end {
                                            let _jp_height_float = j_height_y.0 as f64;
                                            assert!(j_height_y.0 >= 15);
                                            if y <= 7 && j_height_y.0 >= 7 && y >= j_height_y.0
                                                || y >= 8 && j_height_y.0 >= 8 && y <= j_height_y.0
                                            {
                                                val = 1.0;
                                            } else {
                                                val = 0.0
                                            }
                                        }
                                    } else if RADIUS {
                                        // The first 32 bits square (8x8) of indicator I if I > E.
                                        // [(0,0),
                                        //  (1,0), (1,1),
                                        //  (2,0), (2,1), (2,2),
                                        //  (3,0), (3,1), (3,2), (3,3),
                                        //  (4,0), (4,1), (4,2), (4,3), (4,4),
                                        //  (5,0), (5,1), (5,2), (5,3), (5,4), (5,5),
                                        //  (6,0), (6,1), (6,2), (6,3), (6,4), (6,5), (6,6),
                                        //  (7,0), (7,1), (7,2), (7,3), (7,4), (7,5), (7,6), (7,7)]

                                        if i_height_y.1 >= e_height_y.1
                                            && [
                                                (0, 0),
                                                (1, 0),
                                                (1, 1),
                                                (2, 0),
                                                (2, 1),
                                                (2, 2),
                                                (3, 0),
                                                (3, 1),
                                                (3, 2),
                                                (3, 3),
                                                (4, 0),
                                                (4, 1),
                                                (4, 2),
                                                (4, 3),
                                                (4, 4),
                                                (5, 0),
                                                (5, 1),
                                                (5, 2),
                                                (5, 3),
                                                (5, 4),
                                                (5, 5),
                                                (6, 0),
                                                (6, 1),
                                                (6, 2),
                                                (6, 3),
                                                (6, 4),
                                                (6, 5),
                                                (6, 6),
                                                (7, 0),
                                                (7, 1),
                                                (7, 2),
                                                (7, 3),
                                                (7, 4),
                                                (7, 5),
                                                (7, 6),
                                                (7, 7),
                                            ]
                                            .contains(&(y, x))
                                        {
                                            // i_height_y.0 is the max distance from the center of the square
                                            // i_height_y.1 is the max distance from the center of the square
                                            // to the edge of the indicator
                                            if (i_height_y.1) >= i_height_y.0 {
                                                val = 1.0;
                                            } else {
                                                val = 0.0;
                                            }
                                        }
                                        // The first 32 bits square (8x8) of indicator I if I < E.
                                        // [(1,0),
                                        //  (2,0), (2,1),
                                        //  (3,0), (3,1), (3,2),
                                        //  (4,0), (4,1), (4,2), (4,3),
                                        //  (5,0), (5,1), (5,2), (5,3), (5,4),
                                        //  (6,0), (6,1), (6,2), (6,3), (6,4), (6,5),
                                        //  (7,0), (7,1), (7,2), (7,3), (7,4), (7,5), (7,6)]
                                        else if i_height_y.1 < e_height_y.1
                                            && [
                                                (1, 0),
                                                (2, 0),
                                                (2, 1),
                                                (3, 0),
                                                (3, 1),
                                                (3, 2),
                                                (4, 0),
                                                (4, 1),
                                                (4, 2),
                                                (4, 3),
                                                (5, 0),
                                                (5, 1),
                                                (5, 2),
                                                (5, 3),
                                                (5, 4),
                                                (6, 0),
                                                (6, 1),
                                                (6, 2),
                                                (6, 3),
                                                (6, 4),
                                                (6, 5),
                                                (7, 0),
                                                (7, 1),
                                                (7, 2),
                                                (7, 3),
                                                (7, 4),
                                                (7, 5),
                                                (7, 6),
                                            ]
                                            .contains(&(y, x))
                                        {
                                            if (i_height_y.1) >= i_height_y.0 {
                                                val = 1.0;
                                            } else {
                                                val = 0.0;
                                            }
                                        }
                                        // The first 32 bits square (8x8) of indicator E if E > I.
                                        // [(0,0), (0,1), (0,2), (0,3), (0,4), (0,5), (0,6), (0,7),
                                        //  (1,1), (1,2), (1,3), (1,4), (1,5), (1,6), (1,7),
                                        //  (2,2), (2,3), (2,4), (2,5), (2,6), (2,7),
                                        //  (3,3), (3,4), (3,5), (3,6), (3,7),
                                        //  (4,4), (4,5), (4,6), (4,7),
                                        //  (5,5), (5,6), (5,7),
                                        //  (6,6), (6,7),
                                        //  (7,7)]
                                        else if e_height_y.1 > i_height_y.1
                                            && [
                                                (0, 0),
                                                (0, 1),
                                                (0, 2),
                                                (0, 3),
                                                (0, 4),
                                                (0, 5),
                                                (0, 6),
                                                (0, 7),
                                                (1, 1),
                                                (1, 2),
                                                (1, 3),
                                                (1, 4),
                                                (1, 5),
                                                (1, 6),
                                                (1, 7),
                                                (2, 2),
                                                (2, 3),
                                                (2, 4),
                                                (2, 5),
                                                (2, 6),
                                                (2, 7),
                                                (3, 3),
                                                (3, 4),
                                                (3, 5),
                                                (3, 6),
                                                (3, 7),
                                                (4, 4),
                                                (4, 5),
                                                (4, 6),
                                                (4, 7),
                                                (5, 5),
                                                (5, 6),
                                                (5, 7),
                                                (6, 6),
                                                (6, 7),
                                                (7, 7),
                                            ]
                                            .contains(&(y, x))
                                        {
                                            if (e_height_y.1) >= e_height_y.0 {
                                                val = 1.0;
                                            } else {
                                                val = 0.0;
                                            }
                                        }
                                        // The first 32 bits square (8x8) of indicator E if E < I.
                                        // [(0,1), (0,2), (0,3), (0,4), (0,5), (0,6), (0,7),
                                        //  (1,2), (1,3), (1,4), (1,5), (1,6), (1,7),
                                        //  (2,3), (2,4), (2,5), (2,6), (2,7),
                                        //  (3,4), (3,5), (3,6), (3,7),
                                        //  (4,5), (4,6), (4,7),
                                        //  (5,6), (5,7),
                                        //  (6,7)]
                                        else if e_height_y.1 < i_height_y.1
                                            && [
                                                (0, 1),
                                                (0, 2),
                                                (0, 3),
                                                (0, 4),
                                                (0, 5),
                                                (0, 6),
                                                (0, 7),
                                                (1, 2),
                                                (1, 3),
                                                (1, 4),
                                                (1, 5),
                                                (1, 6),
                                                (1, 7),
                                                (2, 3),
                                                (2, 4),
                                                (2, 5),
                                                (2, 6),
                                                (2, 7),
                                                (3, 4),
                                                (3, 5),
                                                (3, 6),
                                                (3, 7),
                                                (4, 5),
                                                (4, 6),
                                                (4, 7),
                                                (5, 6),
                                                (5, 7),
                                                (6, 7),
                                            ]
                                            .contains(&(y, x))
                                        {
                                            if (e_height_y.1) >= e_height_y.0 {
                                                val = 1.0;
                                            } else {
                                                val = 0.0;
                                            }
                                        }
                                        // The first 32 bits square (8x8) of indicator S if S > N.
                                        // [(0,8), (0,9), (0,10), (0,11), (0,12), (0,13), (0,14), (0,15),
                                        //  (1,8), (1,9), (1,10), (1,11), (1,12), (1,13), (1,14),
                                        //  (2,8), (2,9), (2,10), (2,11), (2,12), (2,13),
                                        //  (3,8), (3,9), (3,10), (3,11), (3,12),
                                        //  (4,8), (4,9), (4,10), (4,11),
                                        //  (5,8), (5,9), (5,10),
                                        //  (6,8), (6,9),
                                        //  (7,8)]
                                        else if s_height_y.1 >= n_height_y.1
                                            && [
                                                (0, 8),
                                                (0, 9),
                                                (0, 10),
                                                (0, 11),
                                                (0, 12),
                                                (0, 13),
                                                (0, 14),
                                                (0, 15),
                                                (1, 8),
                                                (1, 9),
                                                (1, 10),
                                                (1, 11),
                                                (1, 12),
                                                (1, 13),
                                                (1, 14),
                                                (2, 8),
                                                (2, 9),
                                                (2, 10),
                                                (2, 11),
                                                (2, 12),
                                                (2, 13),
                                                (3, 8),
                                                (3, 9),
                                                (3, 10),
                                                (3, 11),
                                                (3, 12),
                                                (4, 8),
                                                (4, 9),
                                                (4, 10),
                                                (4, 11),
                                                (5, 8),
                                                (5, 9),
                                                (5, 10),
                                                (6, 8),
                                                (6, 9),
                                                (7, 8),
                                            ]
                                            .contains(&(y, x))
                                        {
                                            if (s_height_y.1) >= s_height_y.0 {
                                                val = 1.0;
                                            } else {
                                                val = 0.0;
                                            }
                                        }
                                        // The first 32 bits square (8x8) of indicator S if S < N.
                                        // [(0,8), (0,9), (0,10), (0,11), (0,12), (0,13), (0,14),
                                        //  (1,8), (1,9), (1,10), (1,11), (1,12), (1,13),
                                        //  (2,8), (2,9), (2,10), (2,11), (2,12),
                                        //  (3,8), (3,9), (3,10), (3,11),
                                        //  (4,8), (4,9), (4,10),
                                        //  (5,8), (5,9),
                                        //  (6,8)]
                                        else if s_height_y.1 < n_height_y.1
                                            && [
                                                (0, 8),
                                                (0, 9),
                                                (0, 10),
                                                (0, 11),
                                                (0, 12),
                                                (0, 13),
                                                (0, 14),
                                                (1, 8),
                                                (1, 9),
                                                (1, 10),
                                                (1, 11),
                                                (1, 12),
                                                (1, 13),
                                                (2, 8),
                                                (2, 9),
                                                (2, 10),
                                                (2, 11),
                                                (2, 12),
                                                (3, 8),
                                                (3, 9),
                                                (3, 10),
                                                (3, 11),
                                                (4, 8),
                                                (4, 9),
                                                (4, 10),
                                                (5, 8),
                                                (5, 9),
                                                (6, 8),
                                            ]
                                            .contains(&(y, x))
                                        {
                                            if (s_height_y.1) >= s_height_y.0 {
                                                val = 1.0;
                                            } else {
                                                val = 0.0;
                                            }
                                        }
                                        // The first 32 bits square (8x8) of indicator N if N > S.
                                        // [ (0,15),
                                        //   (1,14), (1,15),
                                        //   (2,13), (2,14), (2,15),
                                        //   (3,12), (3,13), (3,14), (3,15),
                                        //   (4,11), (4,12), (4,13), (4,14), (4,15),
                                        //   (5,10), (5,11), (5,12), (5,13), (5,14), (5,15),
                                        //   (6,9), (6,10), (6,11), (6,12), (6,13), (6,14), (6,15),
                                        //   (7,8), (7,9), (7,10), (7,11), (7,12), (7,13), (7,14), (7,15)]
                                        else if n_height_y.1 > s_height_y.1
                                            && [
                                                (0, 15),
                                                (1, 14),
                                                (1, 15),
                                                (2, 13),
                                                (2, 14),
                                                (2, 15),
                                                (3, 12),
                                                (3, 13),
                                                (3, 14),
                                                (3, 15),
                                                (4, 11),
                                                (4, 12),
                                                (4, 13),
                                                (4, 14),
                                                (4, 15),
                                                (5, 10),
                                                (5, 11),
                                                (5, 12),
                                                (5, 13),
                                                (5, 14),
                                                (5, 15),
                                                (6, 9),
                                                (6, 10),
                                                (6, 11),
                                                (6, 12),
                                                (6, 13),
                                                (6, 14),
                                                (6, 15),
                                                (7, 8),
                                                (7, 9),
                                                (7, 10),
                                                (7, 11),
                                                (7, 12),
                                                (7, 13),
                                                (7, 14),
                                                (7, 15),
                                            ]
                                            .contains(&(y, x))
                                        {
                                            if (n_height_y.1) >= n_height_y.0 {
                                                val = 1.0;
                                            } else {
                                                val = 0.0;
                                            }
                                        }
                                        // The first 32 bits square (8x8) of indicator N if N < S.
                                        // [(1,15),
                                        //  (2,14), (2,15),
                                        //  (3,13), (3,14), (3,15),
                                        //  (4,12), (4,13), (4,14), (4,15),
                                        //  (5,11), (5,12), (5,13), (5,14), (5,15),
                                        //  (6,10), (6,11), (6,12), (6,13), (6,14), (6,15),
                                        //  (7,9), (7,10), (7,11), (7,12), (7,13), (7,14), (7,15)]
                                        else if n_height_y.1 < s_height_y.1
                                            && [
                                                (1, 15),
                                                (2, 14),
                                                (2, 15),
                                                (3, 13),
                                                (3, 14),
                                                (3, 15),
                                                (4, 12),
                                                (4, 13),
                                                (4, 14),
                                                (4, 15),
                                                (5, 11),
                                                (5, 12),
                                                (5, 13),
                                                (5, 14),
                                                (5, 15),
                                                (6, 10),
                                                (6, 11),
                                                (6, 12),
                                                (6, 13),
                                                (6, 14),
                                                (6, 15),
                                                (7, 9),
                                                (7, 10),
                                                (7, 11),
                                                (7, 12),
                                                (7, 13),
                                                (7, 14),
                                                (7, 15),
                                            ]
                                            .contains(&(y, x))
                                        {
                                            if (n_height_y.1) >= n_height_y.0 {
                                                val = 1.0;
                                            } else {
                                                val = 0.0;
                                            }
                                        }
                                        // The first 32 bits square (8x8) of indicator T if T > F.
                                        // [(8,0), (8,1), (8,2), (8,3), (8,4), (8,5), (8,6), (8,7),
                                        //  (9,0), (9,1), (9,2), (9,3), (9,4), (9,5), (9,6),
                                        //  (10,0), (10,1), (10,2), (10,3), (10,4), (10,5),
                                        //  (11,0), (11,1), (11,2), (11,3), (11,4),
                                        //  (12,0), (12,1), (12,2), (12,3),
                                        //  (13,0), (13,1), (13,2),
                                        //  (14,0), (14,1),
                                        //  (15,0)]
                                        else if t_height_y.1 >= f_height_y.1
                                            && [
                                                (8, 0),
                                                (8, 1),
                                                (8, 2),
                                                (8, 3),
                                                (8, 4),
                                                (8, 5),
                                                (8, 6),
                                                (8, 7),
                                                (9, 0),
                                                (9, 1),
                                                (9, 2),
                                                (9, 3),
                                                (9, 4),
                                                (9, 5),
                                                (9, 6),
                                                (10, 0),
                                                (10, 1),
                                                (10, 2),
                                                (10, 3),
                                                (10, 4),
                                                (10, 5),
                                                (11, 0),
                                                (11, 1),
                                                (11, 2),
                                                (11, 3),
                                                (11, 4),
                                                (12, 0),
                                                (12, 1),
                                                (12, 2),
                                                (12, 3),
                                                (13, 0),
                                                (13, 1),
                                                (13, 2),
                                                (14, 0),
                                                (14, 1),
                                                (15, 0),
                                            ]
                                            .contains(&(y, x))
                                        {
                                            if (t_height_y.1) >= t_height_y.0 {
                                                val = 1.0;
                                            } else {
                                                val = 0.0;
                                            }
                                        }
                                        // The first 32 bits square (8x8) of indicator T if T < F.
                                        // [(8,0), (8,1), (8,2), (8,3), (8,4), (8,5), (8,6),
                                        //  (9,0), (9,1), (9,2), (9,3), (9,4), (9,5),
                                        //  (10,0), (10,1), (10,2), (10,3), (10,4),
                                        //  (11,0), (11,1), (11,2), (11,3),
                                        //  (12,0), (12,1), (12,2),
                                        //  (13,0), (13,1),
                                        //  (14,0)]
                                        else if t_height_y.1 < f_height_y.1
                                            && [
                                                (8, 0),
                                                (8, 1),
                                                (8, 2),
                                                (8, 3),
                                                (8, 4),
                                                (8, 5),
                                                (8, 6),
                                                (9, 0),
                                                (9, 1),
                                                (9, 2),
                                                (9, 3),
                                                (9, 4),
                                                (9, 5),
                                                (10, 0),
                                                (10, 1),
                                                (10, 2),
                                                (10, 3),
                                                (10, 4),
                                                (11, 0),
                                                (11, 1),
                                                (11, 2),
                                                (11, 3),
                                                (12, 0),
                                                (12, 1),
                                                (12, 2),
                                                (13, 0),
                                                (13, 1),
                                                (14, 0),
                                            ]
                                            .contains(&(y, x))
                                        {
                                            if (t_height_y.1) >= t_height_y.0 {
                                                val = 1.0;
                                            } else {
                                                val = 0.0;
                                            }
                                        }
                                        // The first 32 bits square (8x8) of indicator F if F > T.
                                        // [(8,7),
                                        //  (9,6), (9,7),
                                        //  (10,5), (10,6), (10,7),
                                        //  (11,4), (11,5), (11,6), (11,7),
                                        //  (12,3), (12,4), (12,5), (12,6), (12,7),
                                        //  (13,2), (13,3), (13,4), (13,5), (13,6), (13,7),
                                        //  (14,1), (14,2), (14,3), (14,4), (14,5), (14,6), (14,7),
                                        //  (15,0), (15,1), (15,2), (15,3), (15,4), (15,5), (15,6), (15,7)]
                                        else if f_height_y.1 > t_height_y.1
                                            && [
                                                (8, 7),
                                                (9, 6),
                                                (9, 7),
                                                (10, 5),
                                                (10, 6),
                                                (10, 7),
                                                (11, 4),
                                                (11, 5),
                                                (11, 6),
                                                (11, 7),
                                                (12, 3),
                                                (12, 4),
                                                (12, 5),
                                                (12, 6),
                                                (12, 7),
                                                (13, 2),
                                                (13, 3),
                                                (13, 4),
                                                (13, 5),
                                                (13, 6),
                                                (13, 7),
                                                (14, 1),
                                                (14, 2),
                                                (14, 3),
                                                (14, 4),
                                                (14, 5),
                                                (14, 6),
                                                (14, 7),
                                                (15, 0),
                                                (15, 1),
                                                (15, 2),
                                                (15, 3),
                                                (15, 4),
                                                (15, 5),
                                                (15, 6),
                                                (15, 7),
                                            ]
                                            .contains(&(y, x))
                                        {
                                            if (f_height_y.1) >= f_height_y.0 {
                                                val = 1.0;
                                            } else {
                                                val = 0.0;
                                            }
                                        }
                                        // The first 32 bits square (8x8) of indicator F if F < T.
                                        // [(9,7),
                                        //  (10,6), (10,7),
                                        //  (11,5), (11,6), (11,7),
                                        //  (12,4), (12,5), (12,6), (12,7),
                                        //  (13,3), (13,4), (13,5), (13,6), (13,7),
                                        //  (14,2), (14,3), (14,4), (14,5), (14,6), (14,7),
                                        //  (15,1), (15,2), (15,3), (15,4), (15,5), (15,6), (15,7)]
                                        else if f_height_y.1 < t_height_y.1
                                            && [
                                                (9, 7),
                                                (10, 6),
                                                (10, 7),
                                                (11, 5),
                                                (11, 6),
                                                (11, 7),
                                                (12, 4),
                                                (12, 5),
                                                (12, 6),
                                                (12, 7),
                                                (13, 3),
                                                (13, 4),
                                                (13, 5),
                                                (13, 6),
                                                (13, 7),
                                                (14, 2),
                                                (14, 3),
                                                (14, 4),
                                                (14, 5),
                                                (14, 6),
                                                (14, 7),
                                                (15, 1),
                                                (15, 2),
                                                (15, 3),
                                                (15, 4),
                                                (15, 5),
                                                (15, 6),
                                                (15, 7),
                                            ]
                                            .contains(&(y, x))
                                        {
                                            if (f_height_y.1) >= f_height_y.0 {
                                                val = 1.0;
                                            } else {
                                                val = 0.0;
                                            }
                                        }
                                        // The first 32 bits square (8x8) of indicator J if J > P.
                                        // [(8,8),
                                        //  (9,8), (9,9),
                                        //  (10,8), (10,9), (10,10),
                                        //  (11,8), (11,9), (11,10), (11,11),
                                        //  (12,8), (12,9), (12,10), (12,11), (12,12),
                                        //  (13,8), (13,9), (13,10), (13,11), (13,12), (13,13),
                                        //  (14,8), (14,9), (14,10), (14,11), (14,12), (14,13), (14,14),
                                        //  (15,8), (15,9), (15,10), (15,11), (15,12), (15,13), (15,14), (15,15)]
                                        else if j_height_y.1 >= p_height_y.1
                                            && [
                                                (8, 8),
                                                (9, 8),
                                                (9, 9),
                                                (10, 8),
                                                (10, 9),
                                                (10, 10),
                                                (11, 8),
                                                (11, 9),
                                                (11, 10),
                                                (11, 11),
                                                (12, 8),
                                                (12, 9),
                                                (12, 10),
                                                (12, 11),
                                                (12, 12),
                                                (13, 8),
                                                (13, 9),
                                                (13, 10),
                                                (13, 11),
                                                (13, 12),
                                                (13, 13),
                                                (14, 8),
                                                (14, 9),
                                                (14, 10),
                                                (14, 11),
                                                (14, 12),
                                                (14, 13),
                                                (14, 14),
                                                (15, 8),
                                                (15, 9),
                                                (15, 10),
                                                (15, 11),
                                                (15, 12),
                                                (15, 13),
                                                (15, 14),
                                                (15, 15),
                                            ]
                                            .contains(&(y, x))
                                        {
                                            if (j_height_y.1) >= j_height_y.0 {
                                                val = 1.0;
                                            } else {
                                                val = 0.0;
                                            }
                                        }
                                        // The first 32 bits square (8x8) of indicator J if J < P.
                                        //  [(9,8),
                                        //  (10,8), (10,9),
                                        //  (11,8), (11,9), (11,10),
                                        //  (12,8), (12,9), (12,10), (12,11),
                                        //  (13,8), (13,9), (13,10), (13,11), (13,12),
                                        //  (14,8), (14,9), (14,10), (14,11), (14,12), (14,13),
                                        //  (15,8), (15,9), (15,10), (15,11), (15,12), (15,13), (15,14)]
                                        else if j_height_y.1 < p_height_y.1
                                            && [
                                                (9, 8),
                                                (10, 8),
                                                (10, 9),
                                                (11, 8),
                                                (11, 9),
                                                (11, 10),
                                                (12, 8),
                                                (12, 9),
                                                (12, 10),
                                                (12, 11),
                                                (13, 8),
                                                (13, 9),
                                                (13, 10),
                                                (13, 11),
                                                (13, 12),
                                                (14, 8),
                                                (14, 9),
                                                (14, 10),
                                                (14, 11),
                                                (14, 12),
                                                (14, 13),
                                                (15, 8),
                                                (15, 9),
                                                (15, 10),
                                                (15, 11),
                                                (15, 12),
                                                (15, 13),
                                                (15, 14),
                                            ]
                                            .contains(&(y, x))
                                        {
                                            if (j_height_y.1) >= j_height_y.0 {
                                                val = 1.0;
                                            } else {
                                                val = 0.0;
                                            }
                                        }
                                        // The first 32 bits square (8x8) of indicator P if P > J.
                                        // [(8,8), (8,9), (8,10), (8,11), (8,12), (8,13), (8,14), (8,15),
                                        //  (9,9), (9,10), (9,11), (9,12), (9,13), (9,14), (9,15),
                                        //  (10,10), (10,11), (10,12), (10,13), (10,14), (10,15),
                                        //  (11,11), (11,12), (11,13), (11,14), (11,15),
                                        //  (12,12), (12,13), (12,14), (12,15),
                                        //  (13,13), (13,14), (13,15),
                                        //  14,14), (14,15),
                                        //  (15,15)]
                                        else if p_height_y.1 > j_height_y.1
                                            && [
                                                (8, 8),
                                                (8, 9),
                                                (8, 10),
                                                (8, 11),
                                                (8, 12),
                                                (8, 13),
                                                (8, 14),
                                                (8, 15),
                                                (9, 9),
                                                (9, 10),
                                                (9, 11),
                                                (9, 12),
                                                (9, 13),
                                                (9, 14),
                                                (9, 15),
                                                (10, 10),
                                                (10, 11),
                                                (10, 12),
                                                (10, 13),
                                                (10, 14),
                                                (10, 15),
                                                (11, 11),
                                                (11, 12),
                                                (11, 13),
                                                (11, 14),
                                                (11, 15),
                                                (12, 12),
                                                (12, 13),
                                                (12, 14),
                                                (12, 15),
                                                (13, 13),
                                                (13, 14),
                                                (13, 15),
                                                (14, 14),
                                                (14, 15),
                                                (15, 15),
                                            ]
                                            .contains(&(y, x))
                                        {
                                            if (p_height_y.1) >= p_height_y.0 {
                                                val = 1.0;
                                            } else {
                                                val = 0.0;
                                            }
                                        }
                                        // The first 32 bits square (8x8) of indicator P if P < J.
                                        // [(8,9), (8,10), (8,11), (8,12), (8,13), (8,14), (8,15),
                                        //  (9,10), (9,11), (9,12), (9,13), (9,14), (9,15),
                                        // (10,11), (10,12), (10,13), (10,14), (10,15),
                                        //  (11,12), (11,13), (11,14), (11,15),
                                        //  (12,13), (12,14), (12,15),
                                        //  (13,14), (13,15),
                                        // (14,15)]
                                        else if p_height_y.1 < j_height_y.1
                                            && [
                                                (8, 9),
                                                (8, 10),
                                                (8, 11),
                                                (8, 12),
                                                (8, 13),
                                                (8, 14),
                                                (8, 15),
                                                (9, 10),
                                                (9, 11),
                                                (9, 12),
                                                (9, 13),
                                                (9, 14),
                                                (9, 15),
                                                (10, 11),
                                                (10, 12),
                                                (10, 13),
                                                (10, 14),
                                                (10, 15),
                                                (11, 12),
                                                (11, 13),
                                                (11, 14),
                                                (11, 15),
                                                (12, 13),
                                                (12, 14),
                                                (12, 15),
                                                (13, 14),
                                                (13, 15),
                                                (14, 15),
                                            ]
                                            .contains(&(y, x))
                                        {
                                            if (p_height_y.1) >= p_height_y.0 {
                                                val = 1.0;
                                            } else {
                                                val = 0.0;
                                            }
                                        } else {
                                            println!("({}, {})", x, y);
                                            val = 0.0;
                                        }
                                    } else {
                                        // The first 64 bits square (8x8) is the indicator IE.
                                        // [(0,0), (0,1), (0,2), (0,3), (0,4), (0,5), (0,6), (0,7),
                                        //  (1,0), (1,1), (1,2), (1,3), (1,4), (1,5), (1,6), (1,7),
                                        //  (2,0), (2,1), (2,2), (2,3), (2,4), (2,5), (2,6), (2,7),
                                        //  (3,0), (3,1), (3,2), (3,3), (3,4), (3,5), (3,6), (3,7),
                                        //  (4,0), (4,1), (4,2), (4,3), (4,4), (4,5), (4,6), (4,7),
                                        //  (5,0), (5,1), (5,2), (5,3), (5,4), (5,5), (5,6), (5,7),
                                        //  (6,0), (6,1), (6,2), (6,3), (6,4), (6,5), (6,6), (6,7),
                                        //  (7,0), (7,1), (7,2), (7,3), (7,4), (7,5), (7,6), (7,7)]
                                        if x <= 7 && y <= 7 {
                                            if SOFTGRADIENT {
                                                val = soft_gradient(image8[(y, x)], maxes[0]);
                                            } else if HOTENCODED && x <= 3 {
                                                val = if sum_i > sum_e { 1.0 } else { 0.0 };
                                            } else if HOTENCODED && x >= 4 {
                                                val = if sum_e > sum_i { 1.0 } else { 0.0 };
                                            } else {
                                                val = image8[(y, x)] / maxes[0];
                                            }
                                        }
                                        // The second 64 bits square (8x8) is the indicator SN.
                                        // [(0,8), (0,9), (0,10), (0,11), (0,12), (0,13), (0,14), (0,15),
                                        //  (1,8), (1,9), (1,10), (1,11), (1,12), (1,13), (1,14), (1,15),
                                        //  (2,8), (2,9), (2,10), (2,11), (2,12), (2,13), (2,14), (2,15),
                                        //  (3,8), (3,9), (3,10), (3,11), (3,12), (3,13), (3,14), (3,15),
                                        //  (4,8), (4,9), (4,10), (4,11), (4,12), (4,13), (4,14), (4,15),
                                        //  (5,8), (5,9), (5,10), (5,11), (5,12), (5,13), (5,14), (5,15),
                                        //  (6,8), (6,9), (6,10), (6,11), (6,12), (6,13), (6,14), (6,15),
                                        //  (7,8), (7,9), (7,10), (7,11), (7,12), (7,13), (7,14), (7,15)]
                                        else if x >= 8 && y <= 7 {
                                            if SOFTGRADIENT {
                                                val = soft_gradient(image8[(y, x)], maxes[1]);
                                            } else if HOTENCODED && y <= 3 {
                                                val = if sum_s > sum_n { 1.0 } else { 0.0 };
                                            } else if HOTENCODED && y >= 4 {
                                                val = if sum_n > sum_s { 1.0 } else { 0.0 };
                                            } else {
                                                val = image8[(y, x)] / maxes[1];
                                            }
                                        }
                                        // The third 64 bits square (8x8) is the indicator TF.
                                        // [(8,0), (8,1), (8,2), (8,3), (8,4), (8,5), (8,6), (8,7),
                                        //  (9,0), (9,1), (9,2), (9,3), (9,4), (9,5), (9,6), (9,7),
                                        //  (10,0), (10,1), (10,2), (10,3), (10,4), (10,5), (10,6), (10,7),
                                        //  (11,0), (11,1), (11,2), (11,3), (11,4), (11,5), (11,6), (11,7),
                                        //  (12,0), (12,1), (12,2), (12,3), (12,4), (12,5), (12,6), (12,7),
                                        //  (13,0), (13,1), (13,2), (13,3), (13,4), (13,5), (13,6), (13,7),
                                        //  (14,0), (14,1), (14,2), (14,3), (14,4), (14,5), (14,6), (14,7),
                                        //  (15,0), (15,1), (15,2), (15,3), (15,4), (15,5), (15,6), (15,7)]
                                        else if x <= 7 && y >= 8 {
                                            if SOFTGRADIENT {
                                                val = soft_gradient(image8[(y, x)], maxes[2]);
                                            } else if HOTENCODED && y <= 11 {
                                                val = if sum_t > sum_f { 1.0 } else { 0.0 };
                                            } else if HOTENCODED && y >= 12 {
                                                val = if sum_f > sum_t { 1.0 } else { 0.0 };
                                            } else {
                                                val = image8[(y, x)] / maxes[2];
                                            }
                                        }
                                        // The fourth 64 bits square (8x8) is the indicator JP.
                                        // [(8,8), (8,9), (8,10), (8,11), (8,12), (8,13), (8,14), (8,15),
                                        //  (9,8), (9,9), (9,10), (9,11), (9,12), (9,13), (9,14), (9,15),
                                        //  (10,8), (10,9), (10,10), (10,11), (10,12), (10,13), (10,14), (10,15),
                                        //  (11,8), (11,9), (11,10), (11,11), (11,12), (11,13), (11,14), (11,15),
                                        //  (12,8), (12,9), (12,10), (12,11), (12,12), (12,13), (12,14), (12,15),
                                        //  (13,8), (13,9), (13,10), (13,11), (13,12), (13,13), (13,14), (13,15),
                                        //  (14,8), (14,9), (14,10), (14,11), (14,12), (14,13), (14,14), (14,15),
                                        //  (15,8), (15,9), (15,10), (15,11), (15,12), (15,13), (15,14), (15,15)]
                                        else if x >= 8 && y >= 8 {
                                            if SOFTGRADIENT {
                                                val = soft_gradient(image8[(y, x)], maxes[3]);
                                            } else if HOTENCODED && x <= 11 {
                                                val = if sum_j > sum_p { 1.0 } else { 0.0 };
                                            } else if HOTENCODED && x >= 12 {
                                                val = if sum_p > sum_j { 1.0 } else { 0.0 };
                                            } else {
                                                val = image8[(y, x)] / maxes[1];
                                            }
                                        }
                                    }
                                    val
                                });

                            let segment_normalized2: DMatrix<f64> =
                                DMatrix::from_fn(image8.nrows(), image8.ncols(), |y, x| {
                                    const SOFTGRADIENT: bool = false;
                                    const HOTENCODED: bool = false;
                                    const SINEWAVE: bool = false;
                                    const WAVE: bool = false;
                                    const RADIUS: bool = true;
                                    let mut val: f64 = 0.0;
                                    let norm_i = segment_normalized1.slice((0, 0), (8, 4));
                                    let norm_e = segment_normalized1.slice((0, 4), (8, 4));
                                    let norm_s = segment_normalized1.slice((0, 8), (4, 8));
                                    let norm_n = segment_normalized1.slice((4, 8), (4, 8));
                                    let norm_t = segment_normalized1.slice((8, 0), (4, 8));
                                    let norm_f = segment_normalized1.slice((12, 0), (4, 8));
                                    let norm_j = segment_normalized1.slice((8, 8), (8, 4));
                                    let norm_p = segment_normalized1.slice((8, 12), (8, 4));
                                    let _avg_i = norm_i.mean();
                                    let _avg_e = norm_e.mean();
                                    let _avg_s = norm_s.mean();
                                    let _avg_n = norm_n.mean();
                                    let _avg_t = norm_t.mean();
                                    let _avg_f = norm_f.mean();
                                    let _avg_j = norm_j.mean();
                                    let _avg_p = norm_p.mean();
                                    let sum_i = norm_i.sum();
                                    let sum_e = norm_e.sum();
                                    let sum_s = norm_s.sum();
                                    let sum_n = norm_n.sum();
                                    let sum_t = norm_t.sum();
                                    let sum_f = norm_f.sum();
                                    let sum_j = norm_j.sum();
                                    let sum_p = norm_p.sum();
                                    let _max_i = norm_i.max();
                                    let _max_e = norm_e.max();
                                    let _max_s = norm_s.max();
                                    let _max_n = norm_n.max();
                                    let _max_t = norm_t.max();
                                    let _max_f = norm_f.max();
                                    let _max_j = norm_j.max();
                                    let _max_p = norm_p.max();

                                    // let _max =
                                    //     [avg_i, avg_e, avg_s, avg_n, avg_t, avg_f, avg_j, avg_p]
                                    //         .iter()
                                    //         .fold(0.0, |acc: f64, x: &f64| acc.max(*x));

                                    // Calculate the radius at a given quadrant.  The quadrant will be split at a 45 degree
                                    // angle from the center point of either (7, 7), (7, 8), (8, 7), (8, 8) for (y, x).
                                    // The radius is the distance from the center point to the edge of the quadrant.
                                    // a true upper_45 value when x <= 7 and y <= 7
                                    // means the value falls above the 45 degree slope from (7,7) towards (0, 0).
                                    // A false upper_45 value when x <= 7 and y <= 7
                                    // means the value falls below the 45 degree slope from (7,7) towards (0, 0).
                                    // A true upper_45 value when x > 7 and y <= 7
                                    // means the value falls above the 45 degree slope from (7,8) towards (0, 15).
                                    // A false upper_45 value when x > 7 and y <= 7
                                    // means the value falls below the 45 degree slope from (8,7) towards (0, 15).
                                    // A true upper_45 value when x <= 7 and y > 7
                                    // means the value falls above the 45 degree slope from (8,7) towards (15, 0).
                                    // A false upper_45 value when x <= 7 and y > 7
                                    // means the value falls below the 45 degree slope from (8,7) towards (15, 0).
                                    // A true upper_45 value when x > 7 and y > 7
                                    // means the value falls above the 45 degree slope from (8,8) towards (15, 15).
                                    // A false upper_45 value when x > 7 and y > 7
                                    // means the value falls below the 45 degree slope from (8,8) towards (15, 15).
                                    let radial_distance_xy = |x: usize, y: usize, angle: usize| {
                                        let yy = y as f64;
                                        let xx = x as f64;
                                        let mut distance: f64 = 0.0;
                                        if angle == 315 {
                                            let x1 = 7.0;
                                            let y1 = 7.0;
                                            distance =
                                                ((yy - y1).powi(2) + (xx - x1).powi(2)).sqrt();
                                        } else if angle == 45 {
                                            let x1 = 8.0;
                                            let y1 = 7.0;
                                            distance =
                                                ((yy - y1).powi(2) + (xx - x1).powi(2)).sqrt();
                                        } else if angle == 135 {
                                            let x1 = 7.0;
                                            let y1 = 8.0;
                                            distance =
                                                ((yy - y1).powi(2) + (xx - x1).powi(2)).sqrt();
                                        } else if angle == 225 {
                                            let x1 = 8.0;
                                            let y1 = 8.0;
                                            distance =
                                                ((yy - y1).powi(2) + (xx - x1).powi(2)).sqrt();
                                        }
                                        distance
                                    };
                                    let height_at_x =
                                        |x, start, end: usize, x_start, x_end| -> (usize, f64) {
                                            let initial_height = start as f64;
                                            let final_height = end as f64;
                                            let initial_x = x_start as f64;
                                            let final_x = x_end as f64;
                                            let slope = if WAVE {
                                                (final_height - initial_height)
                                                    / (final_x - initial_x)
                                            } else {
                                                0.0
                                            };
                                            let y = slope * (x as f64 - initial_x) + initial_height;
                                            let new_y: usize = if y.round() > 15.0 {
                                                15
                                            } else {
                                                y.round() as usize
                                            };
                                            (new_y, slope)
                                        };
                                    let mut ie_height = if sum_i > sum_e {
                                        8.0 + ((1.0 - ((sum_i / sum_i) * (sum_e / sum_i))) * 16.0)
                                    } else {
                                        7.0 - ((1.0 - ((sum_i / sum_e) * (sum_e / sum_e))) * 16.0)
                                    }
                                        as usize;
                                    let mut sn_height = if sum_s > sum_n {
                                        8.0 + ((1.0 - ((sum_s / sum_s) * (sum_n / sum_s))) * 16.0)
                                    } else {
                                        7.0 - ((1.0 - ((sum_s / sum_n) * (sum_n / sum_n))) * 16.0)
                                    }
                                        as usize;
                                    let mut tf_height = if sum_t > sum_f {
                                        8.0 + ((1.0 - ((sum_t / sum_t) * (sum_f / sum_t))) * 16.0)
                                    } else {
                                        7.0 - ((1.0 - ((sum_t / sum_f) * (sum_f / sum_f))) * 16.0)
                                    }
                                        as usize;
                                    let mut jp_height = if sum_j > sum_p {
                                        8.0 + ((1.0 - ((sum_j / sum_j) * (sum_p / sum_j))) * 16.0)
                                    } else {
                                        7.0 - ((1.0 - ((sum_j / sum_p) * (sum_p / sum_p))) * 16.0)
                                    }
                                        as usize;

                                    if ie_height > 15 {
                                        ie_height = 15;
                                    } else if sn_height > 15 {
                                        sn_height = 15;
                                    } else if tf_height > 15 {
                                        tf_height = 15;
                                    } else if jp_height > 15 {
                                        jp_height = 15;
                                    }
                                    ie_height = ie_height / 16 * 7;
                                    sn_height = sn_height / 16 * 8;
                                    tf_height = tf_height / 16 * 8;
                                    jp_height = jp_height / 16 * 7;

                                    if ie_height > max_height {
                                        max_height = ie_height;
                                    }
                                    if sn_height > max_height {
                                        max_height = sn_height;
                                    }
                                    if tf_height > max_height {
                                        max_height = tf_height;
                                    }
                                    if jp_height > max_height {
                                        max_height = jp_height;
                                    }

                                    let ie_start = 0;
                                    let ie_end = ie_start + 3;
                                    let sn_start = ie_start + 4;
                                    let sn_end = sn_start + 3;
                                    let tf_start = sn_start + 4;
                                    let tf_end = tf_start + 3;
                                    let jp_start = tf_start + 4;
                                    let jp_end = jp_start + 3;
                                    let (
                                        mut i_height_y,
                                        mut e_height_y,
                                        mut s_height_y,
                                        mut n_height_y,
                                        mut t_height_y,
                                        mut f_height_y,
                                        mut j_height_y,
                                        mut p_height_y,
                                    ) = if WAVE {
                                        let (ie_height_y, _ie_slope) = height_at_x(
                                            x, ie_height, sn_height, ie_start, sn_start,
                                        );
                                        let (sn_height_y, _sn_slope) = height_at_x(
                                            x, sn_height, tf_height, sn_start, tf_start,
                                        );
                                        let (tf_height_y, _tf_slope) = height_at_x(
                                            x, tf_height, jp_height, tf_start, jp_start,
                                        );
                                        let (jp_height_y, _jp_slope) =
                                            height_at_x(x, jp_height, jp_height, jp_start, jp_end);
                                        (
                                            (ie_height_y, 0),
                                            (0, 0),
                                            (sn_height_y, 0),
                                            (0, 0),
                                            (tf_height_y, 0),
                                            (0, 0),
                                            (jp_height_y, 0),
                                            (0, 0),
                                        )
                                    } else if RADIUS {
                                        let i_height_y =
                                            radial_distance_xy(x, y, 315).round() as usize;
                                        let e_height_y =
                                            radial_distance_xy(x, y, 315).round() as usize;
                                        let s_height_y =
                                            radial_distance_xy(x, y, 45).round() as usize;
                                        let n_height_y =
                                            radial_distance_xy(x, y, 45).round() as usize;
                                        let t_height_y =
                                            radial_distance_xy(x, y, 135).round() as usize;
                                        let f_height_y =
                                            radial_distance_xy(x, y, 135).round() as usize;
                                        let j_height_y =
                                            radial_distance_xy(x, y, 225).round() as usize;
                                        let p_height_y =
                                            radial_distance_xy(x, y, 225).round() as usize;
                                        let i_y = if sum_i >= sum_e { 7.0 } else { 0.0 };
                                        let e_y = if sum_i < sum_e { 7.0 } else { 0.0 };
                                        let s_y = if sum_s >= sum_n { 8.0 } else { 0.0 };
                                        let n_y = if sum_s < sum_n { 8.0 } else { 0.0 };
                                        let t_y = if sum_t >= sum_f { 8.0 } else { 0.0 };
                                        let f_y = if sum_t < sum_f { 8.0 } else { 0.0 };
                                        let j_y = if sum_j >= sum_p { 7.0 } else { 0.0 };
                                        let p_y = if sum_j < sum_p { 7.0 } else { 0.0 };
                                        (
                                            (i_height_y, i_y as usize),
                                            (e_height_y, e_y as usize),
                                            (s_height_y, s_y as usize),
                                            (n_height_y, n_y as usize),
                                            (t_height_y, t_y as usize),
                                            (f_height_y, f_y as usize),
                                            (j_height_y, j_y as usize),
                                            (p_height_y, p_y as usize),
                                        )
                                    } else {
                                        let (ie_height_y, _ie_slope) =
                                            height_at_x(x, ie_height, ie_height, ie_start, ie_end);
                                        let (sn_height_y, _sn_slope) =
                                            height_at_x(x, sn_height, sn_height, sn_start, sn_end);
                                        let (tf_height_y, _tf_slope) =
                                            height_at_x(x, tf_height, tf_height, tf_start, tf_end);
                                        let (jp_height_y, _jp_slope) =
                                            height_at_x(x, jp_height, jp_height, jp_start, jp_end);
                                        (
                                            (ie_height_y, 0),
                                            (0, 0),
                                            (sn_height_y, 0),
                                            (0, 0),
                                            (tf_height_y, 0),
                                            (0, 0),
                                            (jp_height_y, 0),
                                            (0, 0),
                                        )
                                    };

                                    if i_height_y.0 > 15 {
                                        i_height_y.0 = 15;
                                    }

                                    if s_height_y.0 > 15 {
                                        s_height_y.0 = 15;
                                    }

                                    if t_height_y.0 > 15 {
                                        t_height_y.0 = 15;
                                    }

                                    if j_height_y.0 > 15 {
                                        j_height_y.0 = 15;
                                    }

                                    if RADIUS {
                                        if e_height_y.0 > 15 {
                                            e_height_y.0 = 15;
                                        }
                                        if n_height_y.0 > 15 {
                                            n_height_y.0 = 15;
                                        }
                                        if f_height_y.0 > 15 {
                                            f_height_y.0 = 15;
                                        }
                                        if p_height_y.0 > 15 {
                                            p_height_y.0 = 15;
                                        }
                                    }

                                    if SINEWAVE {
                                        let _y_float = y as f64;
                                        if x >= ie_start && x <= ie_end {
                                            assert!(i_height_y.0 >= 15);
                                            let _ie_height_float = i_height_y.0 as f64;
                                            if (y <= 7 && i_height_y.0 >= 7 && y >= i_height_y.0)
                                                || (y >= 8
                                                    && i_height_y.0 >= 8
                                                    && y <= i_height_y.0)
                                            {
                                                val = 1.0;
                                            } else {
                                                val = 0.0
                                            }
                                        }
                                        if x >= sn_start && x <= sn_end {
                                            assert!(s_height_y.0 >= 15);
                                            let _sn_height_float = s_height_y.0 as f64;
                                            if (y <= 7 && s_height_y.0 >= 7 && y >= s_height_y.0)
                                                || (y >= 8
                                                    && s_height_y.0 >= 8
                                                    && y <= s_height_y.0)
                                            {
                                                val = 1.0;
                                            } else {
                                                val = 0.0
                                            }
                                        }
                                        if x >= tf_start && x <= tf_end {
                                            assert!(t_height_y.0 >= 15);
                                            let _tf_height_float = t_height_y.0 as f64;
                                            if y <= 7 && t_height_y.0 >= 7 && y >= t_height_y.0
                                                || y >= 8 && t_height_y.0 >= 8 && y <= t_height_y.0
                                            {
                                                val = 1.0;
                                            } else {
                                                val = 0.0
                                            }
                                        }
                                        if x >= jp_start && x <= jp_end {
                                            let _jp_height_float = j_height_y.0 as f64;
                                            assert!(j_height_y.0 >= 15);
                                            if y <= 7 && j_height_y.0 >= 7 && y >= j_height_y.0
                                                || y >= 8 && j_height_y.0 >= 8 && y <= j_height_y.0
                                            {
                                                val = 1.0;
                                            } else {
                                                val = 0.0
                                            }
                                        }
                                    } else if RADIUS {
                                        // The first 32 bits square (8x8) of indicator I if I > E.
                                        // [(0,0),
                                        //  (1,0), (1,1),
                                        //  (2,0), (2,1), (2,2),
                                        //  (3,0), (3,1), (3,2), (3,3),
                                        //  (4,0), (4,1), (4,2), (4,3), (4,4),
                                        //  (5,0), (5,1), (5,2), (5,3), (5,4), (5,5),
                                        //  (6,0), (6,1), (6,2), (6,3), (6,4), (6,5), (6,6),
                                        //  (7,0), (7,1), (7,2), (7,3), (7,4), (7,5), (7,6), (7,7)]

                                        if i_height_y.1 >= e_height_y.1
                                            && [
                                                (0, 0),
                                                (1, 0),
                                                (1, 1),
                                                (2, 0),
                                                (2, 1),
                                                (2, 2),
                                                (3, 0),
                                                (3, 1),
                                                (3, 2),
                                                (3, 3),
                                                (4, 0),
                                                (4, 1),
                                                (4, 2),
                                                (4, 3),
                                                (4, 4),
                                                (5, 0),
                                                (5, 1),
                                                (5, 2),
                                                (5, 3),
                                                (5, 4),
                                                (5, 5),
                                                (6, 0),
                                                (6, 1),
                                                (6, 2),
                                                (6, 3),
                                                (6, 4),
                                                (6, 5),
                                                (6, 6),
                                                (7, 0),
                                                (7, 1),
                                                (7, 2),
                                                (7, 3),
                                                (7, 4),
                                                (7, 5),
                                                (7, 6),
                                                (7, 7),
                                            ]
                                            .contains(&(y, x))
                                        {
                                            // i_height_y.0 is the max distance from the center of the square
                                            // i_height_y.1 is the max distance from the center of the square
                                            // to the edge of the indicator
                                            if (i_height_y.1) >= i_height_y.0 {
                                                val = 1.0;
                                            } else {
                                                val = 0.0;
                                            }
                                        }
                                        // The first 32 bits square (8x8) of indicator I if I < E.
                                        // [(1,0),
                                        //  (2,0), (2,1),
                                        //  (3,0), (3,1), (3,2),
                                        //  (4,0), (4,1), (4,2), (4,3),
                                        //  (5,0), (5,1), (5,2), (5,3), (5,4),
                                        //  (6,0), (6,1), (6,2), (6,3), (6,4), (6,5),
                                        //  (7,0), (7,1), (7,2), (7,3), (7,4), (7,5), (7,6)]
                                        else if i_height_y.1 < e_height_y.1
                                            && [
                                                (1, 0),
                                                (2, 0),
                                                (2, 1),
                                                (3, 0),
                                                (3, 1),
                                                (3, 2),
                                                (4, 0),
                                                (4, 1),
                                                (4, 2),
                                                (4, 3),
                                                (5, 0),
                                                (5, 1),
                                                (5, 2),
                                                (5, 3),
                                                (5, 4),
                                                (6, 0),
                                                (6, 1),
                                                (6, 2),
                                                (6, 3),
                                                (6, 4),
                                                (6, 5),
                                                (7, 0),
                                                (7, 1),
                                                (7, 2),
                                                (7, 3),
                                                (7, 4),
                                                (7, 5),
                                                (7, 6),
                                            ]
                                            .contains(&(y, x))
                                        {
                                            if (i_height_y.1) >= i_height_y.0 {
                                                val = 1.0;
                                            } else {
                                                val = 0.0;
                                            }
                                        }
                                        // The first 32 bits square (8x8) of indicator E if E > I.
                                        // [(0,0), (0,1), (0,2), (0,3), (0,4), (0,5), (0,6), (0,7),
                                        //  (1,1), (1,2), (1,3), (1,4), (1,5), (1,6), (1,7),
                                        //  (2,2), (2,3), (2,4), (2,5), (2,6), (2,7),
                                        //  (3,3), (3,4), (3,5), (3,6), (3,7),
                                        //  (4,4), (4,5), (4,6), (4,7),
                                        //  (5,5), (5,6), (5,7),
                                        //  (6,6), (6,7),
                                        //  (7,7)]
                                        else if e_height_y.1 > i_height_y.1
                                            && [
                                                (0, 0),
                                                (0, 1),
                                                (0, 2),
                                                (0, 3),
                                                (0, 4),
                                                (0, 5),
                                                (0, 6),
                                                (0, 7),
                                                (1, 1),
                                                (1, 2),
                                                (1, 3),
                                                (1, 4),
                                                (1, 5),
                                                (1, 6),
                                                (1, 7),
                                                (2, 2),
                                                (2, 3),
                                                (2, 4),
                                                (2, 5),
                                                (2, 6),
                                                (2, 7),
                                                (3, 3),
                                                (3, 4),
                                                (3, 5),
                                                (3, 6),
                                                (3, 7),
                                                (4, 4),
                                                (4, 5),
                                                (4, 6),
                                                (4, 7),
                                                (5, 5),
                                                (5, 6),
                                                (5, 7),
                                                (6, 6),
                                                (6, 7),
                                                (7, 7),
                                            ]
                                            .contains(&(y, x))
                                        {
                                            if (e_height_y.1) >= e_height_y.0 {
                                                val = 1.0;
                                            } else {
                                                val = 0.0;
                                            }
                                        }
                                        // The first 32 bits square (8x8) of indicator E if E < I.
                                        // [(0,1), (0,2), (0,3), (0,4), (0,5), (0,6), (0,7),
                                        //  (1,2), (1,3), (1,4), (1,5), (1,6), (1,7),
                                        //  (2,3), (2,4), (2,5), (2,6), (2,7),
                                        //  (3,4), (3,5), (3,6), (3,7),
                                        //  (4,5), (4,6), (4,7),
                                        //  (5,6), (5,7),
                                        //  (6,7)]
                                        else if e_height_y.1 < i_height_y.1
                                            && [
                                                (0, 1),
                                                (0, 2),
                                                (0, 3),
                                                (0, 4),
                                                (0, 5),
                                                (0, 6),
                                                (0, 7),
                                                (1, 2),
                                                (1, 3),
                                                (1, 4),
                                                (1, 5),
                                                (1, 6),
                                                (1, 7),
                                                (2, 3),
                                                (2, 4),
                                                (2, 5),
                                                (2, 6),
                                                (2, 7),
                                                (3, 4),
                                                (3, 5),
                                                (3, 6),
                                                (3, 7),
                                                (4, 5),
                                                (4, 6),
                                                (4, 7),
                                                (5, 6),
                                                (5, 7),
                                                (6, 7),
                                            ]
                                            .contains(&(y, x))
                                        {
                                            if (e_height_y.1) >= e_height_y.0 {
                                                val = 1.0;
                                            } else {
                                                val = 0.0;
                                            }
                                        }
                                        // The first 32 bits square (8x8) of indicator S if S > N.
                                        // [(0,8), (0,9), (0,10), (0,11), (0,12), (0,13), (0,14), (0,15),
                                        //  (1,8), (1,9), (1,10), (1,11), (1,12), (1,13), (1,14),
                                        //  (2,8), (2,9), (2,10), (2,11), (2,12), (2,13),
                                        //  (3,8), (3,9), (3,10), (3,11), (3,12),
                                        //  (4,8), (4,9), (4,10), (4,11),
                                        //  (5,8), (5,9), (5,10),
                                        //  (6,8), (6,9),
                                        //  (7,8)]
                                        else if s_height_y.1 >= n_height_y.1
                                            && [
                                                (0, 8),
                                                (0, 9),
                                                (0, 10),
                                                (0, 11),
                                                (0, 12),
                                                (0, 13),
                                                (0, 14),
                                                (0, 15),
                                                (1, 8),
                                                (1, 9),
                                                (1, 10),
                                                (1, 11),
                                                (1, 12),
                                                (1, 13),
                                                (1, 14),
                                                (2, 8),
                                                (2, 9),
                                                (2, 10),
                                                (2, 11),
                                                (2, 12),
                                                (2, 13),
                                                (3, 8),
                                                (3, 9),
                                                (3, 10),
                                                (3, 11),
                                                (3, 12),
                                                (4, 8),
                                                (4, 9),
                                                (4, 10),
                                                (4, 11),
                                                (5, 8),
                                                (5, 9),
                                                (5, 10),
                                                (6, 8),
                                                (6, 9),
                                                (7, 8),
                                            ]
                                            .contains(&(y, x))
                                        {
                                            if (s_height_y.1) >= s_height_y.0 {
                                                val = 1.0;
                                            } else {
                                                val = 0.0;
                                            }
                                        }
                                        // The first 32 bits square (8x8) of indicator S if S < N.
                                        // [(0,8), (0,9), (0,10), (0,11), (0,12), (0,13), (0,14),
                                        //  (1,8), (1,9), (1,10), (1,11), (1,12), (1,13),
                                        //  (2,8), (2,9), (2,10), (2,11), (2,12),
                                        //  (3,8), (3,9), (3,10), (3,11),
                                        //  (4,8), (4,9), (4,10),
                                        //  (5,8), (5,9),
                                        //  (6,8)]
                                        else if s_height_y.1 < n_height_y.1
                                            && [
                                                (0, 8),
                                                (0, 9),
                                                (0, 10),
                                                (0, 11),
                                                (0, 12),
                                                (0, 13),
                                                (0, 14),
                                                (1, 8),
                                                (1, 9),
                                                (1, 10),
                                                (1, 11),
                                                (1, 12),
                                                (1, 13),
                                                (2, 8),
                                                (2, 9),
                                                (2, 10),
                                                (2, 11),
                                                (2, 12),
                                                (3, 8),
                                                (3, 9),
                                                (3, 10),
                                                (3, 11),
                                                (4, 8),
                                                (4, 9),
                                                (4, 10),
                                                (5, 8),
                                                (5, 9),
                                                (6, 8),
                                            ]
                                            .contains(&(y, x))
                                        {
                                            if (s_height_y.1) >= s_height_y.0 {
                                                val = 1.0;
                                            } else {
                                                val = 0.0;
                                            }
                                        }
                                        // The first 32 bits square (8x8) of indicator N if N > S.
                                        // [ (0,15),
                                        //   (1,14), (1,15),
                                        //   (2,13), (2,14), (2,15),
                                        //   (3,12), (3,13), (3,14), (3,15),
                                        //   (4,11), (4,12), (4,13), (4,14), (4,15),
                                        //   (5,10), (5,11), (5,12), (5,13), (5,14), (5,15),
                                        //   (6,9), (6,10), (6,11), (6,12), (6,13), (6,14), (6,15),
                                        //   (7,8), (7,9), (7,10), (7,11), (7,12), (7,13), (7,14), (7,15)]
                                        else if n_height_y.1 > s_height_y.1
                                            && [
                                                (0, 15),
                                                (1, 14),
                                                (1, 15),
                                                (2, 13),
                                                (2, 14),
                                                (2, 15),
                                                (3, 12),
                                                (3, 13),
                                                (3, 14),
                                                (3, 15),
                                                (4, 11),
                                                (4, 12),
                                                (4, 13),
                                                (4, 14),
                                                (4, 15),
                                                (5, 10),
                                                (5, 11),
                                                (5, 12),
                                                (5, 13),
                                                (5, 14),
                                                (5, 15),
                                                (6, 9),
                                                (6, 10),
                                                (6, 11),
                                                (6, 12),
                                                (6, 13),
                                                (6, 14),
                                                (6, 15),
                                                (7, 8),
                                                (7, 9),
                                                (7, 10),
                                                (7, 11),
                                                (7, 12),
                                                (7, 13),
                                                (7, 14),
                                                (7, 15),
                                            ]
                                            .contains(&(y, x))
                                        {
                                            if (n_height_y.1) >= n_height_y.0 {
                                                val = 1.0;
                                            } else {
                                                val = 0.0;
                                            }
                                        }
                                        // The first 32 bits square (8x8) of indicator N if N < S.
                                        // [(1,15),
                                        //  (2,14), (2,15),
                                        //  (3,13), (3,14), (3,15),
                                        //  (4,12), (4,13), (4,14), (4,15),
                                        //  (5,11), (5,12), (5,13), (5,14), (5,15),
                                        //  (6,10), (6,11), (6,12), (6,13), (6,14), (6,15),
                                        //  (7,9), (7,10), (7,11), (7,12), (7,13), (7,14), (7,15)]
                                        else if n_height_y.1 < s_height_y.1
                                            && [
                                                (1, 15),
                                                (2, 14),
                                                (2, 15),
                                                (3, 13),
                                                (3, 14),
                                                (3, 15),
                                                (4, 12),
                                                (4, 13),
                                                (4, 14),
                                                (4, 15),
                                                (5, 11),
                                                (5, 12),
                                                (5, 13),
                                                (5, 14),
                                                (5, 15),
                                                (6, 10),
                                                (6, 11),
                                                (6, 12),
                                                (6, 13),
                                                (6, 14),
                                                (6, 15),
                                                (7, 9),
                                                (7, 10),
                                                (7, 11),
                                                (7, 12),
                                                (7, 13),
                                                (7, 14),
                                                (7, 15),
                                            ]
                                            .contains(&(y, x))
                                        {
                                            if (n_height_y.1) >= n_height_y.0 {
                                                val = 1.0;
                                            } else {
                                                val = 0.0;
                                            }
                                        }
                                        // The first 32 bits square (8x8) of indicator T if T > F.
                                        // [(8,0), (8,1), (8,2), (8,3), (8,4), (8,5), (8,6), (8,7),
                                        //  (9,0), (9,1), (9,2), (9,3), (9,4), (9,5), (9,6),
                                        //  (10,0), (10,1), (10,2), (10,3), (10,4), (10,5),
                                        //  (11,0), (11,1), (11,2), (11,3), (11,4),
                                        //  (12,0), (12,1), (12,2), (12,3),
                                        //  (13,0), (13,1), (13,2),
                                        //  (14,0), (14,1),
                                        //  (15,0)]
                                        else if t_height_y.1 >= f_height_y.1
                                            && [
                                                (8, 0),
                                                (8, 1),
                                                (8, 2),
                                                (8, 3),
                                                (8, 4),
                                                (8, 5),
                                                (8, 6),
                                                (8, 7),
                                                (9, 0),
                                                (9, 1),
                                                (9, 2),
                                                (9, 3),
                                                (9, 4),
                                                (9, 5),
                                                (9, 6),
                                                (10, 0),
                                                (10, 1),
                                                (10, 2),
                                                (10, 3),
                                                (10, 4),
                                                (10, 5),
                                                (11, 0),
                                                (11, 1),
                                                (11, 2),
                                                (11, 3),
                                                (11, 4),
                                                (12, 0),
                                                (12, 1),
                                                (12, 2),
                                                (12, 3),
                                                (13, 0),
                                                (13, 1),
                                                (13, 2),
                                                (14, 0),
                                                (14, 1),
                                                (15, 0),
                                            ]
                                            .contains(&(y, x))
                                        {
                                            if (t_height_y.1) >= t_height_y.0 {
                                                val = 1.0;
                                            } else {
                                                val = 0.0;
                                            }
                                        }
                                        // The first 32 bits square (8x8) of indicator T if T < F.
                                        // [(8,0), (8,1), (8,2), (8,3), (8,4), (8,5), (8,6),
                                        //  (9,0), (9,1), (9,2), (9,3), (9,4), (9,5),
                                        //  (10,0), (10,1), (10,2), (10,3), (10,4),
                                        //  (11,0), (11,1), (11,2), (11,3),
                                        //  (12,0), (12,1), (12,2),
                                        //  (13,0), (13,1),
                                        //  (14,0)]
                                        else if t_height_y.1 < f_height_y.1
                                            && [
                                                (8, 0),
                                                (8, 1),
                                                (8, 2),
                                                (8, 3),
                                                (8, 4),
                                                (8, 5),
                                                (8, 6),
                                                (9, 0),
                                                (9, 1),
                                                (9, 2),
                                                (9, 3),
                                                (9, 4),
                                                (9, 5),
                                                (10, 0),
                                                (10, 1),
                                                (10, 2),
                                                (10, 3),
                                                (10, 4),
                                                (11, 0),
                                                (11, 1),
                                                (11, 2),
                                                (11, 3),
                                                (12, 0),
                                                (12, 1),
                                                (12, 2),
                                                (13, 0),
                                                (13, 1),
                                                (14, 0),
                                            ]
                                            .contains(&(y, x))
                                        {
                                            if (t_height_y.1) >= t_height_y.0 {
                                                val = 1.0;
                                            } else {
                                                val = 0.0;
                                            }
                                        }
                                        // The first 32 bits square (8x8) of indicator F if F > T.
                                        // [(8,7),
                                        //  (9,6), (9,7),
                                        //  (10,5), (10,6), (10,7),
                                        //  (11,4), (11,5), (11,6), (11,7),
                                        //  (12,3), (12,4), (12,5), (12,6), (12,7),
                                        //  (13,2), (13,3), (13,4), (13,5), (13,6), (13,7),
                                        //  (14,1), (14,2), (14,3), (14,4), (14,5), (14,6), (14,7),
                                        //  (15,0), (15,1), (15,2), (15,3), (15,4), (15,5), (15,6), (15,7)]
                                        else if f_height_y.1 > t_height_y.1
                                            && [
                                                (8, 7),
                                                (9, 6),
                                                (9, 7),
                                                (10, 5),
                                                (10, 6),
                                                (10, 7),
                                                (11, 4),
                                                (11, 5),
                                                (11, 6),
                                                (11, 7),
                                                (12, 3),
                                                (12, 4),
                                                (12, 5),
                                                (12, 6),
                                                (12, 7),
                                                (13, 2),
                                                (13, 3),
                                                (13, 4),
                                                (13, 5),
                                                (13, 6),
                                                (13, 7),
                                                (14, 1),
                                                (14, 2),
                                                (14, 3),
                                                (14, 4),
                                                (14, 5),
                                                (14, 6),
                                                (14, 7),
                                                (15, 0),
                                                (15, 1),
                                                (15, 2),
                                                (15, 3),
                                                (15, 4),
                                                (15, 5),
                                                (15, 6),
                                                (15, 7),
                                            ]
                                            .contains(&(y, x))
                                        {
                                            if (f_height_y.1) >= f_height_y.0 {
                                                val = 1.0;
                                            } else {
                                                val = 0.0;
                                            }
                                        }
                                        // The first 32 bits square (8x8) of indicator F if F < T.
                                        // [(9,7),
                                        //  (10,6), (10,7),
                                        //  (11,5), (11,6), (11,7),
                                        //  (12,4), (12,5), (12,6), (12,7),
                                        //  (13,3), (13,4), (13,5), (13,6), (13,7),
                                        //  (14,2), (14,3), (14,4), (14,5), (14,6), (14,7),
                                        //  (15,1), (15,2), (15,3), (15,4), (15,5), (15,6), (15,7)]
                                        else if f_height_y.1 < t_height_y.1
                                            && [
                                                (9, 7),
                                                (10, 6),
                                                (10, 7),
                                                (11, 5),
                                                (11, 6),
                                                (11, 7),
                                                (12, 4),
                                                (12, 5),
                                                (12, 6),
                                                (12, 7),
                                                (13, 3),
                                                (13, 4),
                                                (13, 5),
                                                (13, 6),
                                                (13, 7),
                                                (14, 2),
                                                (14, 3),
                                                (14, 4),
                                                (14, 5),
                                                (14, 6),
                                                (14, 7),
                                                (15, 1),
                                                (15, 2),
                                                (15, 3),
                                                (15, 4),
                                                (15, 5),
                                                (15, 6),
                                                (15, 7),
                                            ]
                                            .contains(&(y, x))
                                        {
                                            if (f_height_y.1) >= f_height_y.0 {
                                                val = 1.0;
                                            } else {
                                                val = 0.0;
                                            }
                                        }
                                        // The first 32 bits square (8x8) of indicator J if J > P.
                                        // [(8,8),
                                        //  (9,8), (9,9),
                                        //  (10,8), (10,9), (10,10),
                                        //  (11,8), (11,9), (11,10), (11,11),
                                        //  (12,8), (12,9), (12,10), (12,11), (12,12),
                                        //  (13,8), (13,9), (13,10), (13,11), (13,12), (13,13),
                                        //  (14,8), (14,9), (14,10), (14,11), (14,12), (14,13), (14,14),
                                        //  (15,8), (15,9), (15,10), (15,11), (15,12), (15,13), (15,14), (15,15)]
                                        else if j_height_y.1 >= p_height_y.1
                                            && [
                                                (8, 8),
                                                (9, 8),
                                                (9, 9),
                                                (10, 8),
                                                (10, 9),
                                                (10, 10),
                                                (11, 8),
                                                (11, 9),
                                                (11, 10),
                                                (11, 11),
                                                (12, 8),
                                                (12, 9),
                                                (12, 10),
                                                (12, 11),
                                                (12, 12),
                                                (13, 8),
                                                (13, 9),
                                                (13, 10),
                                                (13, 11),
                                                (13, 12),
                                                (13, 13),
                                                (14, 8),
                                                (14, 9),
                                                (14, 10),
                                                (14, 11),
                                                (14, 12),
                                                (14, 13),
                                                (14, 14),
                                                (15, 8),
                                                (15, 9),
                                                (15, 10),
                                                (15, 11),
                                                (15, 12),
                                                (15, 13),
                                                (15, 14),
                                                (15, 15),
                                            ]
                                            .contains(&(y, x))
                                        {
                                            if (j_height_y.1) >= j_height_y.0 {
                                                val = 1.0;
                                            } else {
                                                val = 0.0;
                                            }
                                        }
                                        // The first 32 bits square (8x8) of indicator J if J < P.
                                        //  [(9,8),
                                        //  (10,8), (10,9),
                                        //  (11,8), (11,9), (11,10),
                                        //  (12,8), (12,9), (12,10), (12,11),
                                        //  (13,8), (13,9), (13,10), (13,11), (13,12),
                                        //  (14,8), (14,9), (14,10), (14,11), (14,12), (14,13),
                                        //  (15,8), (15,9), (15,10), (15,11), (15,12), (15,13), (15,14)]
                                        else if j_height_y.1 < p_height_y.1
                                            && [
                                                (9, 8),
                                                (10, 8),
                                                (10, 9),
                                                (11, 8),
                                                (11, 9),
                                                (11, 10),
                                                (12, 8),
                                                (12, 9),
                                                (12, 10),
                                                (12, 11),
                                                (13, 8),
                                                (13, 9),
                                                (13, 10),
                                                (13, 11),
                                                (13, 12),
                                                (14, 8),
                                                (14, 9),
                                                (14, 10),
                                                (14, 11),
                                                (14, 12),
                                                (14, 13),
                                                (15, 8),
                                                (15, 9),
                                                (15, 10),
                                                (15, 11),
                                                (15, 12),
                                                (15, 13),
                                                (15, 14),
                                            ]
                                            .contains(&(y, x))
                                        {
                                            if (j_height_y.1) >= j_height_y.0 {
                                                val = 1.0;
                                            } else {
                                                val = 0.0;
                                            }
                                        }
                                        // The first 32 bits square (8x8) of indicator P if P > J.
                                        // [(8,8), (8,9), (8,10), (8,11), (8,12), (8,13), (8,14), (8,15),
                                        //  (9,9), (9,10), (9,11), (9,12), (9,13), (9,14), (9,15),
                                        //  (10,10), (10,11), (10,12), (10,13), (10,14), (10,15),
                                        //  (11,11), (11,12), (11,13), (11,14), (11,15),
                                        //  (12,12), (12,13), (12,14), (12,15),
                                        //  (13,13), (13,14), (13,15),
                                        //  14,14), (14,15),
                                        //  (15,15)]
                                        else if p_height_y.1 > j_height_y.1
                                            && [
                                                (8, 8),
                                                (8, 9),
                                                (8, 10),
                                                (8, 11),
                                                (8, 12),
                                                (8, 13),
                                                (8, 14),
                                                (8, 15),
                                                (9, 9),
                                                (9, 10),
                                                (9, 11),
                                                (9, 12),
                                                (9, 13),
                                                (9, 14),
                                                (9, 15),
                                                (10, 10),
                                                (10, 11),
                                                (10, 12),
                                                (10, 13),
                                                (10, 14),
                                                (10, 15),
                                                (11, 11),
                                                (11, 12),
                                                (11, 13),
                                                (11, 14),
                                                (11, 15),
                                                (12, 12),
                                                (12, 13),
                                                (12, 14),
                                                (12, 15),
                                                (13, 13),
                                                (13, 14),
                                                (13, 15),
                                                (14, 14),
                                                (14, 15),
                                                (15, 15),
                                            ]
                                            .contains(&(y, x))
                                        {
                                            if (p_height_y.1) >= p_height_y.0 {
                                                val = 1.0;
                                            } else {
                                                val = 0.0;
                                            }
                                        }
                                        // The first 32 bits square (8x8) of indicator P if P < J.
                                        // [(8,9), (8,10), (8,11), (8,12), (8,13), (8,14), (8,15),
                                        //  (9,10), (9,11), (9,12), (9,13), (9,14), (9,15),
                                        // (10,11), (10,12), (10,13), (10,14), (10,15),
                                        //  (11,12), (11,13), (11,14), (11,15),
                                        //  (12,13), (12,14), (12,15),
                                        //  (13,14), (13,15),
                                        // (14,15)]
                                        else if p_height_y.1 < j_height_y.1
                                            && [
                                                (8, 9),
                                                (8, 10),
                                                (8, 11),
                                                (8, 12),
                                                (8, 13),
                                                (8, 14),
                                                (8, 15),
                                                (9, 10),
                                                (9, 11),
                                                (9, 12),
                                                (9, 13),
                                                (9, 14),
                                                (9, 15),
                                                (10, 11),
                                                (10, 12),
                                                (10, 13),
                                                (10, 14),
                                                (10, 15),
                                                (11, 12),
                                                (11, 13),
                                                (11, 14),
                                                (11, 15),
                                                (12, 13),
                                                (12, 14),
                                                (12, 15),
                                                (13, 14),
                                                (13, 15),
                                                (14, 15),
                                            ]
                                            .contains(&(y, x))
                                        {
                                            if (p_height_y.1) >= p_height_y.0 {
                                                val = 1.0;
                                            } else {
                                                val = 0.0;
                                            }
                                        } else {
                                            println!("({}, {})", x, y);
                                            val = 0.0;
                                        }
                                    } else {
                                        // The first 64 bits square (8x8) is the indicator IE.
                                        // [(0,0), (0,1), (0,2), (0,3), (0,4), (0,5), (0,6), (0,7),
                                        //  (1,0), (1,1), (1,2), (1,3), (1,4), (1,5), (1,6), (1,7),
                                        //  (2,0), (2,1), (2,2), (2,3), (2,4), (2,5), (2,6), (2,7),
                                        //  (3,0), (3,1), (3,2), (3,3), (3,4), (3,5), (3,6), (3,7),
                                        //  (4,0), (4,1), (4,2), (4,3), (4,4), (4,5), (4,6), (4,7),
                                        //  (5,0), (5,1), (5,2), (5,3), (5,4), (5,5), (5,6), (5,7),
                                        //  (6,0), (6,1), (6,2), (6,3), (6,4), (6,5), (6,6), (6,7),
                                        //  (7,0), (7,1), (7,2), (7,3), (7,4), (7,5), (7,6), (7,7)]
                                        if x <= 7 && y <= 7 {
                                            if SOFTGRADIENT {
                                                val = soft_gradient(image8[(y, x)], maxes[0]);
                                            } else if HOTENCODED && x <= 3 {
                                                val = if sum_i > sum_e { 1.0 } else { 0.0 };
                                            } else if HOTENCODED && x >= 4 {
                                                val = if sum_e > sum_i { 1.0 } else { 0.0 };
                                            } else {
                                                val = image8[(y, x)] / maxes[0];
                                            }
                                        }
                                        // The second 64 bits square (8x8) is the indicator SN.
                                        // [(0,8), (0,9), (0,10), (0,11), (0,12), (0,13), (0,14), (0,15),
                                        //  (1,8), (1,9), (1,10), (1,11), (1,12), (1,13), (1,14), (1,15),
                                        //  (2,8), (2,9), (2,10), (2,11), (2,12), (2,13), (2,14), (2,15),
                                        //  (3,8), (3,9), (3,10), (3,11), (3,12), (3,13), (3,14), (3,15),
                                        //  (4,8), (4,9), (4,10), (4,11), (4,12), (4,13), (4,14), (4,15),
                                        //  (5,8), (5,9), (5,10), (5,11), (5,12), (5,13), (5,14), (5,15),
                                        //  (6,8), (6,9), (6,10), (6,11), (6,12), (6,13), (6,14), (6,15),
                                        //  (7,8), (7,9), (7,10), (7,11), (7,12), (7,13), (7,14), (7,15)]
                                        else if x >= 8 && y <= 7 {
                                            if SOFTGRADIENT {
                                                val = soft_gradient(image8[(y, x)], maxes[1]);
                                            } else if HOTENCODED && y <= 3 {
                                                val = if sum_s > sum_n { 1.0 } else { 0.0 };
                                            } else if HOTENCODED && y >= 4 {
                                                val = if sum_n > sum_s { 1.0 } else { 0.0 };
                                            } else {
                                                val = image8[(y, x)] / maxes[1];
                                            }
                                        }
                                        // The third 64 bits square (8x8) is the indicator TF.
                                        // [(8,0), (8,1), (8,2), (8,3), (8,4), (8,5), (8,6), (8,7),
                                        //  (9,0), (9,1), (9,2), (9,3), (9,4), (9,5), (9,6), (9,7),
                                        //  (10,0), (10,1), (10,2), (10,3), (10,4), (10,5), (10,6), (10,7),
                                        //  (11,0), (11,1), (11,2), (11,3), (11,4), (11,5), (11,6), (11,7),
                                        //  (12,0), (12,1), (12,2), (12,3), (12,4), (12,5), (12,6), (12,7),
                                        //  (13,0), (13,1), (13,2), (13,3), (13,4), (13,5), (13,6), (13,7),
                                        //  (14,0), (14,1), (14,2), (14,3), (14,4), (14,5), (14,6), (14,7),
                                        //  (15,0), (15,1), (15,2), (15,3), (15,4), (15,5), (15,6), (15,7)]
                                        else if x <= 7 && y >= 8 {
                                            if SOFTGRADIENT {
                                                val = soft_gradient(image8[(y, x)], maxes[2]);
                                            } else if HOTENCODED && y <= 11 {
                                                val = if sum_t > sum_f { 1.0 } else { 0.0 };
                                            } else if HOTENCODED && y >= 12 {
                                                val = if sum_f > sum_t { 1.0 } else { 0.0 };
                                            } else {
                                                val = image8[(y, x)] / maxes[2];
                                            }
                                        }
                                        // The fourth 64 bits square (8x8) is the indicator JP.
                                        // [(8,8), (8,9), (8,10), (8,11), (8,12), (8,13), (8,14), (8,15),
                                        //  (9,8), (9,9), (9,10), (9,11), (9,12), (9,13), (9,14), (9,15),
                                        //  (10,8), (10,9), (10,10), (10,11), (10,12), (10,13), (10,14), (10,15),
                                        //  (11,8), (11,9), (11,10), (11,11), (11,12), (11,13), (11,14), (11,15),
                                        //  (12,8), (12,9), (12,10), (12,11), (12,12), (12,13), (12,14), (12,15),
                                        //  (13,8), (13,9), (13,10), (13,11), (13,12), (13,13), (13,14), (13,15),
                                        //  (14,8), (14,9), (14,10), (14,11), (14,12), (14,13), (14,14), (14,15),
                                        //  (15,8), (15,9), (15,10), (15,11), (15,12), (15,13), (15,14), (15,15)]
                                        else if x >= 8 && y >= 8 {
                                            if SOFTGRADIENT {
                                                val = soft_gradient(image8[(y, x)], maxes[3]);
                                            } else if HOTENCODED && x <= 11 {
                                                val = if sum_j > sum_p { 1.0 } else { 0.0 };
                                            } else if HOTENCODED && x >= 12 {
                                                val = if sum_p > sum_j { 1.0 } else { 0.0 };
                                            } else {
                                                val = image8[(y, x)] / maxes[1];
                                            }
                                        }
                                    }
                                    val
                                });

                            assert_eq!(indicator::ISFJ & indicator::mb_flag::E, 0);
                            assert_eq!(indicator::ISFJ & indicator::mb_flag::N, 0);
                            assert_eq!(indicator::ISFJ & indicator::mb_flag::T, 0);
                            assert_eq!(indicator::ISFJ & indicator::mb_flag::P, 0);

                            // Verify classifier is valid.
                            const DEBUG: bool = false;
                            let label = classifiers[i];
                            let valid;
                            let mut invalid_reason: String = String::from("");
                            // Check for Valid I.
                            if segment_normalized2[(7, 0)] == 1.0
                                && label & indicator::mb_flag::I == 0
                            {
                                valid = false;
                                invalid_reason.push_str("I");
                            }
                            // Check for Valid E.
                            else if segment_normalized2[(0, 7)] == 1.0
                                && label & indicator::mb_flag::E == 0
                            {
                                valid = false;
                                invalid_reason.push_str("E");
                            }
                            // Check for Valid S.
                            else if segment_normalized2[(0, 8)] == 1.0
                                && label & indicator::mb_flag::S == 0
                            {
                                valid = false;
                                invalid_reason.push_str("S");
                            }
                            // Check for Valid N.
                            else if segment_normalized2[(7, 15)] == 1.0
                                && label & indicator::mb_flag::N == 0
                            {
                                valid = false;
                                invalid_reason.push_str("N");
                            }
                            // Check for Valid T.
                            else if segment_normalized2[(8, 0)] == 1.0
                                && label & indicator::mb_flag::T == 0
                            {
                                valid = false;
                                invalid_reason.push_str("T");
                            }
                            // Check for Valid F.
                            else if segment_normalized2[(15, 7)] == 1.0
                                && label & indicator::mb_flag::F == 0
                            {
                                valid = false;
                                invalid_reason.push_str("F");
                            }
                            // Check for Valid J.
                            else if segment_normalized2[(15, 8)] == 1.0
                                && label & indicator::mb_flag::J == 0
                            {
                                valid = false;
                                invalid_reason.push_str("J");
                            }
                            // Check for Valid P.
                            else if segment_normalized2[(8, 15)] == 1.0
                                && label & indicator::mb_flag::P == 0
                            {
                                valid = false;
                                invalid_reason.push_str("P");
                            } else {
                                valid = true;
                                if DEBUG {
                                    println!("Valid: {}", MBTI { indicator: label }.to_string());
                                }
                            }

                            if valid {
                                // Flatten the 2D DMatrix into a 1D Vector.
                                let mut flattened: Vec<f64> = Vec::new();
                                let mut max = segment_normalized2.max();
                                for y in 0..16 {
                                    for x in 0..16 {
                                        let val = segment_normalized2[(y, x)];
                                        flattened.push(val);
                                    }
                                }
                                assert_eq!(flattened.len(), 256);
                                if max == 0.0 {
                                    max = 1.0;
                                }
                                let normalized: Vec<f64> = {
                                    let mut normalized: Vec<f64> = Vec::new();
                                    for x in 0..256 {
                                        let val = flattened[x];
                                        let decimal_val: Decimal =
                                            Decimal::from_f64(val / max).unwrap();
                                        normalized.push(decimal_val.round_dp(2).to_f64().unwrap());
                                    }
                                    normalized
                                };
                                classifiers1.push(label);
                                heatmap.push(normalized);
                            } else {
                                if DEBUG {
                                    println!(
                                        "Invalid: {}, Valid: {}",
                                        invalid_reason,
                                        MBTI { indicator: label }.to_string()
                                    );
                                }
                            }

                            if i + 1 % 25000 == 0 {
                                println!("i: {} of {}", i, corpus.nrows());
                            }
                        }
                        println!("Max height {}", max_height);
                        println!("Max {:?}", max);
                        println!("Max avg {:?}", max_avg);
                        println!("Max sum {:?}", max_sum);
                        // TODO I put corpus 16 in here.
                        println!("Saving heatmap...");
                        let f = std::fs::OpenOptions::new()
                            .write(true)
                            .create(true)
                            .truncate(true)
                            .open(path_heatmap);
                        let bytes = bincode::serialize(&heatmap).unwrap();
                        f.and_then(|mut f| f.write_all(&bytes))
                            .expect("Failed to write heatmap");
                        println!("Done.");
                        println!("Saving classifiers1...");
                        let f = std::fs::OpenOptions::new()
                            .write(true)
                            .create(true)
                            .truncate(true)
                            .open(path_classifiers1);
                        let bytes = bincode::serialize(&classifiers1).unwrap();
                        f.and_then(|mut f| f.write_all(&bytes))
                            .expect("Failed to write classifiers1");
                        println!("Done.");
                    } else {
                        let f = std::fs::File::open(path_heatmap).unwrap();
                        heatmap = bincode::deserialize_from(f).unwrap();
                        // let f = std::fs::File::open(path_classifiers1).unwrap();
                        // classifiers1 = bincode::deserialize_from(f).unwrap();
                    }
                    heatmap
                };

                let mut barchart: Vec<Vec<f64>> = Vec::new();
                for i in 0..corpus.nrows() {
                    let mut row: Vec<f64> = Vec::new();
                    let mut max_overall: f64 = 0.0;
                    for x in 0..16 {
                        let val_overall: f64 = overall[(i, x)];
                        max_overall = if val_overall > max_overall {
                            val_overall
                        } else {
                            max_overall
                        };
                    }
                    for y in 0..16 {
                        let height: f64 = 16.0 - y as f64;
                        let axis = 8.0;
                        for x in 0..16 {
                            let val_overall: f64 = overall[(i, x)];
                            let fill = val_overall / max_overall * 16.0;
                            if height >= axis && axis + (fill / 2.0) >= height
                                || height < axis && axis - (fill / 2.0) <= height
                            {
                                row.push(1.0);
                            } else {
                                row.push(0.0);
                            }
                        }
                    }
                    assert_eq!(row.len(), 256);
                    barchart.push(row);
                }

                // let mut heatchart: Vec<Vec<f64>> = Vec::new();
                // for i in 0..corpus.nrows() {
                //     let mut row: Vec<f64> = Vec::new();
                //     for y in 0..personality_freqs.nrows() {
                //         let _height: f64 = 16.0 - y as f64;
                //         for x in 0..corpus.ncols() {
                //             if x % 3 == 0 {
                //                 // it spaces the values out by inserting 4 empty columns to get to 16.
                //                 row.push(0.0);
                //             }
                //             let term_freq = overall_terms[(i, x)][y];
                //             let temp: Decimal = Decimal::from_f64(term_freq).unwrap();
                //             row.push(temp.round_dp(2).to_f64().unwrap());
                //         }
                //     }
                //     assert_eq!(row.len(), 256);
                //     heatchart.push(row);
                // }

                // let mut normalized: Vec<Vec<f64>> = Vec::new();
                // for i in 0..heatchart.len() {
                //     let mut row: Vec<f64> = Vec::new();
                //     let mut max = 0.0;
                //     for j in 0..heatchart[i].len() {
                //         let val = heatchart[i][j];
                //         if val > max {
                //             max = val;
                //         }
                //     }
                //     for j in 0..heatchart[i].len() {
                //         let val = heatchart[i][j];
                //         let decimal_val: Decimal = Decimal::from_f64(val / max).unwrap();
                //         row.push(decimal_val.round_dp(2).to_f64().unwrap());
                //     }
                //     normalized.push(row);
                // }

                println!("Saving normalized visual signal...");
                let path_normalized = &Path::new("./normalized_visual_signal.bincode");
                let f = std::fs::OpenOptions::new()
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .open(path_normalized);
                let bytes = bincode::serialize(&heatmap).unwrap();
                f.and_then(|mut f| f.write_all(&bytes))
                    .expect("Failed to write visual signal");

                let _personality_frequency = {
                    let path_personality_frequency = &Path::new("./personality_frequency.bincode");
                    let mut personality_frequency: Vec<Vec<f64>>;
                    if !path_personality_frequency.exists() {
                        println!("Calculating personality frequency...");
                        personality_frequency = Vec::new();
                        for i in 0..corpus.nrows() {
                            let mut row: Vec<f64> = Vec::new();
                            let mut max = 0.0;
                            for j in 0..corpus.ncols() {
                                for n in 0..16 {
                                    let val = overall_terms[(i, j)][n];
                                    if val > max {
                                        max = val;
                                    }
                                }
                            }
                            for x in 0..corpus.ncols() {
                                for y in 0..16 {
                                    let val = overall_terms[(i, x)][y];
                                    let decimal_val: Decimal =
                                        Decimal::from_f64(val / max).unwrap();
                                    row.push(decimal_val.round_dp(2).to_f64().unwrap());
                                }
                            }
                            personality_frequency.push(row);
                        }
                        println!("Saving personality frequency...");
                        let f = std::fs::OpenOptions::new()
                            .write(true)
                            .create(true)
                            .truncate(true)
                            .open(path_personality_frequency);
                        let bytes = bincode::serialize(&personality_frequency).unwrap();
                        f.and_then(|mut f| f.write_all(&bytes))
                            .expect("Failed to write personality frequency");
                        println!("Done.");
                    } else {
                        let f = std::fs::File::open(path_personality_frequency).unwrap();
                        personality_frequency = bincode::deserialize_from(f).unwrap();
                    }
                    personality_frequency
                };

                let mut corpus_tf_idf: Vec<Vec<f64>> = Vec::new();
                for x in 0..tf_idf.nrows() {
                    let mut row: Vec<f64> = Vec::new();
                    for y in 0..tf_idf.ncols() {
                        let val = tf_idf[(x, y)];
                        let decimal_val: Decimal = Decimal::from_f64(val).unwrap();
                        row.push(decimal_val.round_dp(4).to_f64().unwrap());
                    }
                    corpus_tf_idf.push(row);
                }
                // TODO revert this override with the normalized_visual_signal.bincode
                println!("Saving tf_idf...");
                let f = std::fs::OpenOptions::new()
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .open(path);
                // let tf_idf_bytes = bincode::serialize(&corpus_tf_idf).unwrap();
                let tf_idf_bytes = bincode::serialize(&heatmap).unwrap();
                f.and_then(|mut f| f.write_all(&tf_idf_bytes))
                    .expect("Failed to write tf_idf");
                // corpus_tf_idf
                heatmap
            }
        };

        // Create f64 matrices for y_set.
        // let classifiers: Vec<u8> = classifiers.iter().map(|y| *y).collect();
        // Load classifiers1
        let f = std::fs::File::open("./classifiers1.bincode").unwrap();
        let classifiers1: Vec<u8> = bincode::deserialize_from(f).unwrap();
        let classifiers = classifiers1;

        let corpus_bytes = bincode::serialize(&tf_idf).expect("Can not serialize the matrix");
        File::create("corpus.bincode")
            .and_then(|mut f| f.write_all(&corpus_bytes))
            .expect("Can not persist corpus");
        let classifiers_bytes: Vec<u8> =
            bincode::serialize(&classifiers).expect("Can not serialize the matrix");
        File::create("classifiers.bincode")
            .and_then(|mut f| f.write_all(&classifiers_bytes))
            .expect("Can not persist classifiers");
        (tf_idf, classifiers)
    } else {
        println!("Loading x and y matrices...");
        let mut x_buf = Vec::new();
        let mut y_buf = Vec::new();
        std::fs::File::open("corpus.bincode")
            .expect("Can not open corpus")
            .read_to_end(&mut x_buf)
            .unwrap();
        std::fs::File::open("classifiers.bincode")
            .expect("Can not open classifiers")
            .read_to_end(&mut y_buf)
            .unwrap();
        let corpus: Vec<Vec<f64>> = bincode::deserialize(&x_buf).unwrap();
        let classifiers: Vec<u8> = bincode::deserialize(&y_buf).unwrap();
        (corpus, classifiers)
    }
}

fn tally(classifiers: &Vec<u8>) {
    println!("Tallying...");
    println!(
        "count ISTJ: {}",
        classifiers
            .iter()
            .filter(|&m| *m == indicator::ISTJ)
            .count()
    );
    println!(
        "count ISFJ: {}",
        classifiers
            .iter()
            .filter(|&m| *m == indicator::ISFJ)
            .count()
    );
    println!(
        "count INFJ: {}",
        classifiers
            .iter()
            .filter(|&m| *m == indicator::INFJ)
            .count()
    );
    println!(
        "count INTJ: {}",
        classifiers
            .iter()
            .filter(|&m| *m == indicator::INTJ)
            .count()
    );
    println!(
        "count ISTP: {}",
        classifiers
            .iter()
            .filter(|&m| *m == indicator::ISTP)
            .count()
    );
    println!(
        "count ISFP: {}",
        classifiers
            .iter()
            .filter(|&m| *m == indicator::ISFP)
            .count()
    );
    println!(
        "count INFP: {}",
        classifiers
            .iter()
            .filter(|&m| *m == indicator::INFP)
            .count()
    );
    println!(
        "count INTP: {}",
        classifiers
            .iter()
            .filter(|&m| *m == indicator::INTP)
            .count()
    );
    println!(
        "count ESTP: {}",
        classifiers
            .iter()
            .filter(|&m| *m == indicator::ESTP)
            .count()
    );
    println!(
        "count ESFP: {}",
        classifiers
            .iter()
            .filter(|&m| *m == indicator::ESFP)
            .count()
    );
    println!(
        "count ENFP: {}",
        classifiers
            .iter()
            .filter(|&m| *m == indicator::ENFP)
            .count()
    );
    println!(
        "count ENTP: {}",
        classifiers
            .iter()
            .filter(|&m| *m == indicator::ENTP)
            .count()
    );
    println!(
        "count ESTJ: {}",
        classifiers
            .iter()
            .filter(|&m| *m == indicator::ESTJ)
            .count()
    );
    println!(
        "count ESFJ: {}",
        classifiers
            .iter()
            .filter(|&m| *m == indicator::ESFJ)
            .count()
    );
    println!(
        "count ENFJ: {}",
        classifiers
            .iter()
            .filter(|&m| *m == indicator::ENFJ)
            .count()
    );
    println!(
        "count ENTJ: {}",
        classifiers
            .iter()
            .filter(|&m| *m == indicator::ENTJ)
            .count()
    );
}

fn train(corpus: &Vec<Vec<f64>>, classifiers: &Vec<u8>, member_id: &str) {
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
                    let bytes_rf = bincode::serialize(&rf).unwrap();
                    File::create(format!("mbti_rf__{}.model", member_id))
                        .and_then(|mut f| f.write_all(&bytes_rf))
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
                        let f = std::fs::OpenOptions::new()
                            .write(true)
                            .create(true)
                            .truncate(true)
                            .open(path_model);
                        let bytes = bincode::serialize(&svm).unwrap();
                        f.and_then(|mut f| f.write_all(&bytes))
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

fn build_sets(
    corpus: &Vec<Vec<f64>>,
    classifiers: &Vec<u8>,
    leaf_a: u8,
    leaf_b: u8,
) -> (Vec<Vec<f64>>, Vec<u8>, Vec<u8>) {
    let mut left_features: Vec<Vec<f64>> = Vec::new();
    let mut right_features: Vec<Vec<f64>> = Vec::new();
    let mut left_labels: Vec<u8> = Vec::new();
    let mut right_labels: Vec<u8> = Vec::new();
    let mut original_labels: Vec<u8> = Vec::new();
    let (side_index, min_sample_size) = {
        let mut acc: Vec<usize> = Vec::new();
        for (i, y) in classifiers.iter().enumerate() {
            let left = *y & leaf_a != 0u8;
            let right = *y & leaf_b != 0u8;
            if left {
                left_features.push(corpus[i].clone());
                left_labels.push(0u8);
                acc.push(0);
                original_labels.push(*y);
            } else if right {
                right_features.push(corpus[i].clone());
                right_labels.push(1u8);
                acc.push(1);
                original_labels.push(*y);
            } else {
                panic!("Invalid classifier");
            }
        }
        assert!(left_features.len() + right_features.len() == acc.len());
        assert!(left_features.len() == left_labels.len());
        assert!(right_features.len() == right_labels.len());
        (acc, left_features.len().min(right_features.len()))
    };
    println!("Min sample size: {}", min_sample_size);
    println!("All samples included");
    // println!("Limited to {} samples", min_sample_size);
    let new_corpus = {
        let mut acc: Vec<Vec<f64>> = Vec::new();
        let mut n: usize = 0;
        let mut m: usize = 0;
        for (_, y) in side_index.iter().enumerate() {
            // Comment below conditoinal to allow All samples.
            if *y == 0 {
                // if n < min_sample_size {
                acc.push(left_features.get(n).unwrap().clone());
                n += 1;
                // }
            } else {
                // if m < min_sample_size {
                acc.push(right_features.get(m).unwrap().clone());
                m += 1;
                // }
            }
        }
        // assert!(acc.len() == min_sample_size * 2);
        acc
    };
    let new_labels: Vec<u8> = {
        let mut acc: Vec<u8> = Vec::new();
        let mut n: usize = 0;
        let mut m: usize = 0;
        for (_, y) in side_index.iter().enumerate() {
            if *y == 0 {
                // if n < min_sample_size {
                acc.push(left_labels.get(n).unwrap().clone());
                n += 1;
                // }
            } else {
                // if m < min_sample_size {
                acc.push(right_labels.get(m).unwrap().clone());
                m += 1;
                // }
            }
        }
        // assert!(acc.len() == min_sample_size * 2);
        acc
    };
    (new_corpus, new_labels, original_labels)
}

fn main() -> Result<(), Error> {
    let training_set: Vec<Sample> = load_data();
    let (data_x, data_y) = normalize(&training_set);
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

    let mut map = Map::new();
    let mut data = [
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    ];
    assert_eq!(
        0b01000000 ^ 0b00010000 ^ 0b00000100 ^ 0b00000001,
        0b01010101
    );
    let normalized_path = Path::new("./normalized_visual_signal.bincode");
    let mut buf = Vec::new();
    File::open(normalized_path)
        .unwrap()
        .read_to_end(&mut buf)
        .expect("Unable to read file");

    let save_sample_to_dir = |sample: Vec<f64>, idx: usize, id: usize, dir: &str| {
        let mut path = PathBuf::from(dir);
        path.push(format!("{}/{}.png", idx, id));
        let segment = DMatrix::from_vec(16, 16, sample.clone());
        // Construct a new by repeated calls to the supplied closure.
        let img = ImageBuffer::from_fn(16, 16, |y, x| {
            if segment[(y as usize, x as usize)] == 0.0 {
                image::Luma([255u8])
            } else {
                image::Luma([0u8])
            }
        });

        // Save the image as ./visual_signal/[idx]/[id].png.
        img.save(path)
            .expect("The sample image could not be saved to ./visual_signal.");
    };
    let visual_signal: Vec<Vec<f64>> = bincode::deserialize(&buf).unwrap();
    for (i, sample) in visual_signal.iter().enumerate() {
        let segment: DMatrix<f64> = DMatrix::from_fn(16, 16, |y, x| sample.get((y * 16 + x) as usize));
        let mbti = MBTI {
            indicator: data_y[i],
        };
        let label: String = mbti.to_string();
        // Verify classifier is valid.
        const DEBUG: bool = false;
        let valid;
        let mut invalid_reason: String = String::from("");
        // Check for Valid I.
        if segment[(7, 0)] == 1.0 && mbti.indicator & indicator::mb_flag::I == 0 {
            valid = false;
            invalid_reason.push_str("I");
        }
        // Check for Valid E.
        else if segment[(0, 7)] == 1.0 && mbti.indicator & indicator::mb_flag::E == 0 {
            valid = false;
            invalid_reason.push_str("E");
        }
        // Check for Valid S.
        else if segment[(0, 8)] == 1.0 && mbti.indicator & indicator::mb_flag::S == 0 {
            valid = false;
            invalid_reason.push_str("S");
        }
        // Check for Valid N.
        else if segment[(7, 15)] == 1.0 && mbti.indicator & indicator::mb_flag::N == 0 {
            valid = false;
            invalid_reason.push_str("N");
        }
        // Check for Valid T.
        else if segment[(8, 0)] == 1.0 && mbti.indicator & indicator::mb_flag::T == 0 {
            valid = false;
            invalid_reason.push_str("T");
        }
        // Check for Valid F.
        else if segment[(15, 7)] == 1.0 && mbti.indicator & indicator::mb_flag::F == 0 {
            valid = false;
            invalid_reason.push_str("F");
        }
        // Check for Valid J.
        else if segment[(15, 8)] == 1.0 && mbti.indicator & indicator::mb_flag::J == 0 {
            valid = false;
            invalid_reason.push_str("J");
        }
        // Check for Valid P.
        else if segment[(8, 15)] == 1.0 && mbti.indicator & indicator::mb_flag::P == 0 {
            valid = false;
            invalid_reason.push_str("P");
        } else {
            valid = true;
            if DEBUG {
                println!(
                    "Valid: {}",
                    MBTI {
                        indicator: mbti.indicator
                    }
                    .to_string()
                );
            }
        }
        if !valid {
            println!("Some indicators are invalid. {}", invalid_reason);
        }
        let mut row: Vec<Value> = Vec::new();
        for (_j, feature) in sample.iter().enumerate() {
            let term: Value = Value::Number(serde_json::Number::from_f64(*feature).unwrap());
            row.push(term);
        }

        let idx = match label.as_str() {
            "ESFJ" => 0,  // "ESFJ",
            "ESFP" => 1,  // "ESFP",
            "ESTJ" => 2,  // "ESTJ",
            "ESTP" => 3,  // "ESTP",
            "ENFJ" => 4,  // "ENFJ",
            "ENFP" => 5,  // "ENFP",
            "ENTJ" => 6,  // "ENTJ",
            "ENTP" => 7,  // "ENTP",
            "ISFJ" => 8,  // "ISFJ",
            "ISFP" => 9,  // "ISFP",
            "ISTJ" => 10, // "ISTJ",
            "ISTP" => 11, // "ISTP",
            "INFJ" => 12, // "INFJ",
            "INFP" => 13, // "INFP",
            "INTJ" => 14, // "INTJ",
            "INTP" => 15, // "INTP",
            _ => panic!("Invalid label"),
        };
        data[idx].push(Value::Array(row));
        save_sample_to_dir(sample.clone(), idx, i, "./visual_signal");
    }
    for n in 0..16 {
        let label = match n {
            0 => "ESFJ",
            1 => "ESFP",
            2 => "ESTJ",
            3 => "ESTP",
            4 => "ENFJ",
            5 => "ENFP",
            6 => "ENTJ",
            7 => "ENTP",
            8 => "ISFJ",
            9 => "ISFP",
            10 => "ISTJ",
            11 => "ISTP",
            12 => "INFJ",
            13 => "INFP",
            14 => "INTJ",
            15 => "INTP",
            _ => panic!("Invalid label"),
        };
        println!("label: {}, data {}", label.to_string(), n);
        map.insert(n.to_string(), Value::Array(data[n].clone()));
    }
    let obj = Value::Object(map);
    let mbti_json_path = Path::new("./mbti_samples.json");
    if !mbti_json_path.exists() {
        let f = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(mbti_json_path);
        let stringify = serde_json::to_string(&obj).unwrap();
        let bytes = stringify.as_bytes();
        f.and_then(|mut f| f.write_all(&bytes))
            .expect("Failed to write samples");
    }

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

    let model_rf_all_path = Path::new("mbti_rf__ALL.model");
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
        let mut buf: Vec<u8> = Vec::new();
        File::open(format!("mbti_rf__{}.model", "ALL"))
            .and_then(|mut f| f.read_to_end(&mut buf))
            .expect("Can not load model");
        bincode::deserialize(&buf).expect("Can not deserialize the model")
    };

    let mut models: Vec<SVC<f64, DenseMatrix<f64>, LinearKernel>> = Vec::new();
    for i in 0..ensemble.len() {
        let _final_model = (i + 1) * 3;
        let filename = format!("./mbti_svm__{}.model", trees[i]);

        let path_model = Path::new(&filename);
        if !path_model.exists() {
            println! {"Training SVM model for {}", trees[i]};
            let ensemble_x = ensemble[i].0.clone();
            let ensemble_y = ensemble[i].1.clone();
            // Potentially limit ensemble training size here with below:
            // train(
            //     &ensemble_x[0..2000].to_vec(),
            //     &ensemble_y[0..2000].to_vec(),
            //     &trees[i],
            // );
            train(&ensemble_x, &ensemble_y, &trees[i]);
            let svm: SVC<f64, DenseMatrix<f64>, LinearKernel> = {
                let mut buf: Vec<u8> = Vec::new();
                File::open(path_model)
                    .and_then(|mut f| f.read_to_end(&mut buf))
                    .expect("Can not load model");
                bincode::deserialize(&buf).expect("Can not deserialize the model")
            };
            models.push(svm);
        } else {
            println!("Loading svm {} model...", trees[i]);
            let svm: SVC<f64, DenseMatrix<f64>, LinearKernel> = {
                let mut buf: Vec<u8> = Vec::new();
                File::open(path_model)
                    .and_then(|mut f| f.read_to_end(&mut buf))
                    .expect("Can not load model");
                bincode::deserialize(&buf).expect("Can not deserialize the model")
            };
            models.push(svm);
        }
    }
    // Get predictions for ensemble
    let mut ensemble_pred: [Vec<f64>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let mut ensemble_y_test: [Vec<f64>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    for (i, model) in models.iter().enumerate() {
        // Also limit the data set here.
        // let x_validate = DenseMatrix::from_2d_vec(&ensemble[i].0[0..5000].to_vec());
        let x_validate = DenseMatrix::from_2d_vec(&ensemble[i].0);
        // let y_validate = ensemble[i].1[0..5000]
        let y_validate = ensemble[i]
            .1
            .to_vec()
            .iter()
            .map(|y| *y as f64)
            .collect::<Vec<f64>>();
        let (_x_train, x_test, _y_train, y_test) =
            train_test_split(&x_validate, &y_validate, 0.2, false);
        ensemble_pred[i] = model.predict(&x_test).unwrap();
        // let original_label_ie = ensemble[0].2[4000..5000].to_vec()
        let original_label_ie = ensemble[0]
            .2
            .iter()
            .map(|x| (MBTI { indicator: *x }).to_string().chars().nth(0).unwrap())
            .collect::<Vec<char>>();
        // let original_label_ns: Vec<char> = ensemble[1].2[4000..5000].to_vec()
        let original_label_ns: Vec<char> = ensemble[1]
            .2
            .iter()
            .map(|x| (MBTI { indicator: *x }).to_string().chars().nth(1).unwrap())
            .collect::<Vec<char>>();
        // let original_label_tf: Vec<char> = ensemble[2].2[4000..5000].to_vec()
        let original_label_tf: Vec<char> = ensemble[2]
            .2
            .iter()
            .map(|x| (MBTI { indicator: *x }).to_string().chars().nth(2).unwrap())
            .collect::<Vec<char>>();
        // let original_label_jp: Vec<char> = ensemble[3].2[4000..5000].to_vec()
        let original_label_jp: Vec<char> = ensemble[3]
            .2
            .iter()
            .map(|x| (MBTI { indicator: *x }).to_string().chars().nth(3).unwrap())
            .collect::<Vec<char>>();
        let original_labels: Vec<f64> = {
            let mut labels: Vec<f64> = Vec::new();
            for i in 0..original_label_ie.len() {
                let label = format!(
                    "{}{}{}{}",
                    original_label_ie[i],
                    original_label_ns[i],
                    original_label_tf[i],
                    original_label_jp[i]
                );
                labels.push(MBTI::from_string(&label).indicator as f64);
            }
            labels
        };
        ensemble_y_test[i] = original_labels;

        println!(
            "{} accuracy: {}",
            trees[i],
            accuracy(&y_test, &ensemble_pred[i])
        );
        println!("MSE: {}", mean_squared_error(&y_test, &ensemble_pred[i]));
    }
    let mut svm_ensemble_y_pred: Vec<f64> = Vec::new();
    let mut svm_ensembly_y_test: Vec<f64> = Vec::new();
    assert!(ensemble_pred.len() == 4);
    for i in 0..ensemble_pred[0].len() {
        let mut mbti: u8 = 0u8;
        for j in 0..ensemble.len() {
            let prediction = ensemble_pred[j].get(i);
            // trees = ["IE", "NS", "TF", "JP"]; (Defined above)
            let tree = trees[j];
            let leaf_a: char = tree.chars().nth(0).unwrap();
            let leaf_b: char = tree.chars().nth(1).unwrap();
            let low: bool = prediction == 0f64;
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
        assert!(
            (ensemble_y_test[0][i] == ensemble_y_test[1][i])
                && (ensemble_y_test[1][i] == ensemble_y_test[2][i]
                    && (ensemble_y_test[2][i] == ensemble_y_test[3][i]))
        );
        svm_ensembly_y_test.push(ensemble_y_test[0][i]);
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
        println!("Prediction, Actual, Diff, Variance, Mean Variance, Correct");
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
                "{},       {},   {}     {:.2},     {:.2}           {}",
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
    let (_x_train, x_test, _y_train, y_test) = train_test_split(&x, &y, 0.22222222222, true);
    println!(
        "Generic Random Forest accuracy: {}",
        accuracy(&y_test, &rf.predict(&x_test).unwrap())
    );
    sample_report(&y_test, &rf.predict(&x_test).unwrap());

    // Print the ensemble report.
    println!(
        "Ensemble Support Vector Machine accuracy: {}",
        accuracy(&svm_ensembly_y_test, &svm_ensemble_y_pred)
    );
    println!(
        "MSE: {}",
        mean_squared_error(&svm_ensembly_y_test, &svm_ensemble_y_pred)
    );
    sample_report(&svm_ensembly_y_test, &svm_ensemble_y_pred);

    Ok(())
}
