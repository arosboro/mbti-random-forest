use csv::Error;
use rand::seq::SliceRandom;
use rand::thread_rng;
// use rand::Rng;
// use rust_bert::pipelines::pos_tagging::POSConfig;
use serde::{Deserialize, Serialize};
use smartcore::svm::svc::{SVCParameters, SVC};
use smartcore::svm::{Kernels, LinearKernel};
use smartcore::tree::decision_tree_classifier::SplitCriterion;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{Read, Write};
use serde_json::{Map, Value};
use std::path::Path;
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
use rust_stemmers::{Algorithm, Stemmer};
use smartcore::metrics::{accuracy, mean_squared_error, roc_auc_score};
use smartcore::model_selection::{cross_validate, train_test_split, KFold};
use stopwords::{Language, Spark, Stopwords};
use vtext::tokenize::*;
use rust_decimal::prelude::*;

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
type Features = (f64, f64);

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
            let mut samples: Vec<Sample> = Vec::new();
            let mut reader = csv::Reader::from_path(csv_target).unwrap();
            let expressions = [
                Regex::new(r"https?://[a-zA-Z0-9_%./]+\??(([a-z0-9%]+=[a-z0-9%]+&?)+)?").unwrap(),
                Regex::new(r"[^a-zA-Z0-9 ]").unwrap(),
                Regex::new(r"\s+").unwrap(),
            ];
            for row in reader.deserialize::<Row>() {
                match row {
                    Ok(row) => {
                        // Split each row on the delimiter "|||" and collect into a vector
                        for post_str in row.posts.split("|||") {
                            // Turn a string into a vector of lematized tokens.  @see tokenize
                            let lemmas: Lemmas = tokenize(post_str, &expressions);
                            if lemmas.len() > 0 {
                                // println!("{}: {}", &row.r#type, tokens.join(" "));
                                let sample: Sample = Sample {
                                    label: MBTI::from_string(&row.r#type),
                                    lemmas,
                                };
                                samples.push(sample)
                            }
                        }
                    }
                    Err(e) => println!("Error: {}", e),
                }
            }
            // Save samples
            let f = std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(path);
            let samples_bytes = bincode::serialize(&samples).unwrap();
            f.and_then(|mut f| f.write_all(&samples_bytes))
                .expect("Failed to write samples");
            samples
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
                    esfj_freq,
                    esfp_freq,
                    estj_freq,
                    estp_freq,
                    enfj_freq,
                    enfp_freq,
                    entj_freq,
                    entp_freq,
                    isfj_freq,
                    isfp_freq,
                    istj_freq,
                    istp_freq,
                    infj_freq,
                    infp_freq,
                    intj_freq,
                    intp_freq,
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
                        if score > 5.0 {
                            println!("Score: {} unexpectedly high with term {}.", score, corpus[(i, j)]);
                        }
                        (score, dictionary[&corpus[(i, j)]])
                    });
                // Normalizing makes the values too tiny.
                // let max = tf_idf.max();
                // let tf_idf_normal: DMatrix<f64> = DMatrix::from_fn(tf_idf.nrows(), tf_idf.ncols(), |i, j| tf_idf[(i, j)] / max);
                println!("tf_idf: {} seconds", start.elapsed().as_secs());
                // Convert tf_idf to a Vec<Vec<f64>>.
                let df1: DMatrix<f64> = DMatrix::from_fn(tf_idf.nrows(), tf_idf.ncols(), |i, j| tf_idf[(i, j)].0);
                let max = df1.max();
                let min = df1.min();
                println!("max: {}", max);
                println!("min: {}", min);
                let tf_idf_normal: DMatrix<f64> = DMatrix::from_fn(tf_idf.nrows(), tf_idf.ncols(), |i, j| df1[(i, j)] / max);
                let df2: DMatrix<f64> = DMatrix::from_fn(tf_idf.nrows(), tf_idf.ncols(), |i, j| tf_idf[(i, j)].1 / dictionary.len() as f64);
                // let df3: DMatrix<f64> = DMatrix::from_fn(tf_idf.nrows(), tf_idf.ncols(), |i, j| tf_idf[(i, j)].2);
                // let grid: [DMatrix<f64>; 3] = [df1, df2, df3];

                let mut corpus_tf_idf: Vec<Vec<f64>> = Vec::new();
                for i in 0..df1.nrows() {
                    let mut row: Vec<f64> = Vec::new();
                    for y in 0..12 {
                        for x in 0..12 {
                            let height: f64 = 12.0 - y as f64;
                            let freq: f64 = df1[(i, x)];
                            let size: f64 = 12.0;
                            let _d_height: Decimal = Decimal::from_f64(height).unwrap();
                            let _d_freq: Decimal = Decimal::from_f64(tf_idf_normal[(i, x)]).unwrap();
                            let _id: f64 = df2[(i, x)];
                            assert!(max < size);
                            if height >= 0.0 && freq * max > height {
                                row.push(1.0);
                            }
                            else {
                                row.push(0.0);
                            }
                        }
                    }
                    assert_eq!(row.len(), 144);
                    corpus_tf_idf.push(row);
                }

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
                        overall = DMatrix::from_fn(corpus.nrows(), 16, |i, j| {
                            if i % 250 == 0 {
                                println!("overall: {} of {}", i, corpus.nrows());
                            }
                            corpus.row(i).iter().fold(0.0, |mut acc: f64, term| {
                                let mut val: f64 = 0.0;
                                if personality_freq[j].contains_key(term) {
                                    val = personality_freq[j][term];
                                }
                                acc += (val / overall_freq[term]) as f64;
                                acc
                            })
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
                        f.write_all(&overall_bytes).expect("Failed to write overall");
                        println!("Saved overall frequencies for each classification.");
                    }
                    else {
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
                        if i % 250 == 0 {
                            println!("overall_terms: {} of {}", i, corpus.nrows());
                        }
                        let term = &corpus[(i, j)];
                        // a tuple containing 16 values, one for each personality type.
                        let mut val: [f64; 16] = [0.0; 16];
                        for x in 0..16 {
                            if personality_freq[x].contains_key(term) {
                                val[x] = (personality_freq[x][term] / overall_freq[term]) as f64;
                            }
                            else {
                                val[x] = 0.0;
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
                let _heatmap: Vec<Vec<f64>> = {
                    let path_heatmap = Path::new("./heatmap.bincode");
                    let mut heatmap: Vec<Vec<f64>>;
                    if !path_heatmap.exists() {
                        // Calculate the 256 bit grid.
                        heatmap = Vec::new();
                        let max_overall = overall.max();
                        for i in 0..corpus.nrows() {
                            let image: DMatrix<f64> = DMatrix::from_fn(16, 16, |y, x| {
                                let n: usize;
                                // The first personality type is plotted like so:
                                // [(0,0), (0,1), (0,2), (0,3),
                                //  (1,0), (1,1), (1,2), (1,3),
                                //  (2,0), (2,1), (2,2), (2,3),
                                //  (3,0), (3,1), (3,2), (3,3)]
                                if x <= 3 && y <= 3 {
                                    n = 0;
                                }
                                // The second personality type is plotted like so:
                                // [(4,0), (4,1), (4,2), (4,3),
                                //  (5,0), (5,1), (5,2), (5,3),
                                //  (6,0), (6,1), (6,2), (6,3),
                                //  (7,0), (7,1), (7,2), (7,3)]
                                else if x >= 4 && x <= 7 && y <= 3 {
                                    n = 1;
                                }
                                // The third personality type is plotted like so:
                                // [(8,0), (8,1), (8,2), (8,3),
                                //  (9,0), (9,1), (9,2), (9,3),
                                //  (10,0), (10,1), (10,2), (10,3),
                                //  (11,0), (11,1), (11,2), (11,3)]
                                else if x >= 8 && x <= 11 && y <= 3 {
                                    n = 2;
                                }
                                // The fourth personality type is plotted like so:
                                // [(12,0), (12,1), (12,2), (12,3),
                                //  (13,0), (13,1), (13,2), (13,3),
                                //  (14,0), (14,1), (14,2), (14,3),
                                //  (15,0), (15,1), (15,2), (15,3)]
                                else if x >= 12 && x <= 15 && y <= 3 {
                                    n = 3;
                                }
                                // The fifth personality type is plotted like so:
                                // [(0,4), (0,5), (0,6), (0,7),
                                //  (1,4), (1,5), (1,6), (1,7),
                                //  (2,4), (2,5), (2,6), (2,7),
                                //  (3,4), (3,5), (3,6), (3,7)]
                                else if x <= 3 && y >= 4 && y <= 7 {
                                    n = 4;
                                }
                                // The sixth personality type is plotted like so:
                                // [(4,4), (4,5), (4,6), (4,7),
                                //  (5,4), (5,5), (5,6), (5,7),
                                //  (6,4), (6,5), (6,6), (6,7),
                                //  (7,4), (7,5), (7,6), (7,7)]
                                else if x >= 4 && x <= 7 && y >= 4 && y <= 7 {
                                    n = 5;
                                }
                                // The seventh personality type is plotted like so:
                                // [(8,4), (8,5), (8,6), (8,7),
                                //  (9,4), (9,5), (9,6), (9,7),
                                //  (10,4), (10,5), (10,6), (10,7),
                                //  (11,4), (11,5), (11,6), (11,7)]
                                else if x >= 8 && x <= 11 && y >= 4 && y <= 7 {
                                    n = 6;
                                }
                                // The eighth personality type is plotted like so:
                                // [(12,4), (12,5), (12,6), (12,7),
                                //  (13,4), (13,5), (13,6), (13,7),
                                //  (14,4), (14,5), (14,6), (14,7),
                                //  (15,4), (15,5), (15,6), (15,7)]
                                else if x >= 12 && x <= 15 && y >= 4 && y <= 7 {
                                    n = 7;
                                }
                                // The ninth personality type is plotted like so:
                                // [(0,8), (0,9), (0,10), (0,11),
                                //  (1,8), (1,9), (1,10), (1,11),
                                //  (2,8), (2,9), (2,10), (2,11),
                                //  (3,8), (3,9), (3,10), (3,11)]
                                else if x <= 3 && y >= 8 && y <= 11 {
                                    n = 8;
                                }
                                // The tenth personality type is plotted like so:
                                // [(4,8), (4,9), (4,10), (4,11),
                                //  (5,8), (5,9), (5,10), (5,11),
                                //  (6,8), (6,9), (6,10), (6,11),
                                //  (7,8), (7,9), (7,10), (7,11)]
                                else if x >= 4 && x <= 7 && y >= 8 && y <= 11 {
                                    n = 9;
                                }
                                // The eleventh personality type is plotted like so:
                                // [(8,8), (8,9), (8,10), (8,11),
                                //  (9,8), (9,9), (9,10), (9,11),
                                //  (10,8), (10,9), (10,10), (10,11),
                                //  (11,8), (11,9), (11,10), (11,11)]
                                else if x >= 8 && x <= 11 && y >= 8 && y <= 11 {
                                    n = 10;
                                }
                                // The twelfth personality type is plotted like so:
                                // [(12,8), (12,9), (12,10), (12,11),
                                //  (13,8), (13,9), (13,10), (13,11),
                                //  (14,8), (14,9), (14,10), (14,11),
                                //  (15,8), (15,9), (15,10), (15,11)]
                                else if x >= 12 && x <= 15 && y >= 8 && y <= 11 {
                                    n = 11;
                                }
                                // The thirteenth personality type is plotted like so:
                                // [(0,12), (0,13), (0,14), (0,15),
                                //  (1,12), (1,13), (1,14), (1,15),
                                //  (2,12), (2,13), (2,14), (2,15),
                                //  (3,12), (3,13), (3,14), (3,15)]
                                else if x <= 3 && y >= 12 && y <= 15 {
                                    n = 12;
                                }
                                // The fourteenth personality type is plotted like so:
                                // [(4,12), (4,13), (4,14), (4,15),
                                //  (5,12), (5,13), (5,14), (5,15),
                                //  (6,12), (6,13), (6,14), (6,15),
                                //  (7,12), (7,13), (7,14), (7,15)]
                                else if x >= 4 && x <= 7 && y >= 12 && y <= 15 {
                                    n = 13;
                                }
                                // The fifteenth personality type is plotted like so:
                                // [(8,12), (8,13), (8,14), (8,15),
                                //  (9,12), (9,13), (9,14), (9,15),
                                //  (10,12), (10,13), (10,14), (10,15),
                                //  (11,12), (11,13), (11,14), (11,15)]
                                else if x >= 8 && x <= 11 && y >= 12 && y <= 15 {
                                    n = 14;
                                }
                                // The sixteenth personality type is plotted like so:
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
                            // Flatten the 2D DMatrix into a 1D Vector.
                            let mut flattened: Vec<f64> = Vec::new();
                            let mut max = 0.0;
                            for i in 0..16 {
                                for j in 0..16 {
                                    let val = image[(i, j)];
                                    if val > max {
                                        max = val;
                                    }
                                    flattened.push(val);
                                }
                            }
                            assert_eq!(flattened.len(), 256);
                            let normalized: Vec<f64> = {
                                let mut normalized: Vec<f64> = Vec::new();
                                for i in 0..256 {
                                    let val = flattened[i];
                                    let decimal_val: Decimal = Decimal::from_f64(val / max).unwrap();
                                    normalized.push(decimal_val.round_dp(2).to_f64().unwrap());
                                }
                                normalized
                            };
                            heatmap.push(normalized);
                            if i % 250 == 0 {
                                println!("i: {} of {}", i, corpus.nrows());
                            }
                        }
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
                    }
                    else {
                        let f = std::fs::File::open(path_heatmap).unwrap();
                        heatmap = bincode::deserialize_from(f).unwrap();
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
                            if height >= axis && axis + (fill / 2.0) >= height ||
                                height < axis && axis - (fill / 2.0) <= height {
                                row.push(1.0);
                            }
                            else {
                                row.push(0.0);
                            }
                        }
                    }
                    assert_eq!(row.len(), 256);
                    barchart.push(row);
                }

                let mut heatchart: Vec<Vec<f64>> = Vec::new();
                for i in 0..corpus.nrows() {
                    let mut row: Vec<f64> = Vec::new();
                    for y in 0..16 {
                        let _height: f64 = 16.0 - y as f64;
                        for x in 0..12 {
                            if x % 3 == 0 {
                                // it spaces the values out by inserting 4 empty columns to get to 16.
                                row.push(0.0);
                            }
                            let term_freq = overall_terms[(i, x)][y];
                            let temp: Decimal = Decimal::from_f64(term_freq).unwrap();
                            row.push(temp.round_dp(2).to_f64().unwrap());
                        }
                    }
                    assert_eq!(row.len(), 256);
                    heatchart.push(row);
                }

                let mut normalized: Vec<Vec<f64>> = Vec::new();
                for i in 0..heatchart.len() {
                    let mut row: Vec<f64> = Vec::new();
                    let mut max = 0.0;
                    for j in 0..heatchart[i].len() {
                        let val = heatchart[i][j];
                        if val > max {
                            max = val;
                        }
                    }
                    for j in 0..heatchart[i].len() {
                        let val = heatchart[i][j];
                        let decimal_val: Decimal = Decimal::from_f64(val / max).unwrap();
                        row.push(decimal_val.round_dp(2).to_f64().unwrap());
                    }
                    normalized.push(row);
                }

                println!("Saving tf_idf...");
                let f = std::fs::OpenOptions::new()
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .open(path);
                let tf_idf_bytes = bincode::serialize(&normalized).unwrap();
                f.and_then(|mut f| f.write_all(&tf_idf_bytes))
                    .expect("Failed to write tf_idf");
                normalized
            }
        };

        // Create f64 matrices for y_set.
        let classifiers: Vec<u8> = classifiers.iter().map(|y| *y).collect();

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

    let x: DenseMatrix<f64> = DenseMatrix::new(
        corpus.len(),
        corpus[0].len(),
        corpus.clone().into_iter().flatten().collect(),
    );
    // These are our target class labels
    let y: Vec<f64> = classifiers.into_iter().map(|x| *x as f64).collect();
    // Split bag into training/test (80%/20%)
    // let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true);

    let rf_ml_wrapper =
        |x: &DenseMatrix<f64>, y: &Vec<f64>, parameters: RandomForestClassifierParameters| {
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
                println!("Random forest accuracy: {}, MSE: {}", accuracy(y, &y_hat_rf), mean_squared_error(y, &y_hat_rf));
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
                    // Evaluate
                    let y_hat_svm: Vec<f64> = svm.predict(x).unwrap();
                    println!("SVM accuracy: {}, MSE: {}, AUC SVM: {}", accuracy(y, &y_hat_svm), mean_squared_error(y, &y_hat_svm), roc_auc_score(y, &y_hat_svm));
                }
                // Evaluate
                let y_hat_svm: Vec<f64> = svm.predict(x).unwrap();
                println!("SVM accuracy: {}, MSE: {}, AUC SVM: {}", accuracy(y, &y_hat_svm), mean_squared_error(y, &y_hat_svm), roc_auc_score(y, &y_hat_svm));
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
        max_depth: Some(8),
        /// The minimum number of samples required to be at a leaf node. See [Decision Tree Classifier](../../tree/decision_tree_classifier/index.html)
        min_samples_leaf: 1,
        /// The minimum number of samples required to split an internal node. See [Decision Tree Classifier](../../tree/decision_tree_classifier/index.html)
        min_samples_split: 2,
        /// The number of trees in the forest.
        n_trees: 100,
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
            .with_c(1.0)
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
            tweaked_svm_params,
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
) -> (Vec<Vec<f64>>, Vec<u8>) {
    let mut left_features: Vec<Vec<f64>> = Vec::new();
    let mut right_features: Vec<Vec<f64>> = Vec::new();
    let mut left_labels: Vec<u8> = Vec::new();
    let mut right_labels: Vec<u8> = Vec::new();
    let (side_index, min_sample_size) = {
        let mut acc: Vec<usize> = Vec::new();
        for (i, y) in classifiers.iter().enumerate() {
            let left = *y & leaf_a != 0u8;
            let right = *y & leaf_b != 0u8;
            if left {
                left_features.push(corpus[i].clone());
                left_labels.push(0u8);
                acc.push(0);
            } else if right {
                right_features.push(corpus[i].clone());
                right_labels.push(1u8);
                acc.push(1);
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
    (new_corpus, new_labels)
}

fn main() -> Result<(), Error> {
    let training_set: Vec<Sample> = load_data();
    let (corpus, classifiers) = normalize(&training_set);
    let mut map = Map::new();
    let mut data = [
        Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), Vec::new(), Vec::new(), Vec::new(),
    ];
    assert_eq!(0b01000000 ^ 0b00010000 ^ 0b00000100 ^ 0b00000001, 0b01010101);
    for (i, sample) in corpus.iter().enumerate() {
        let mbti = MBTI { indicator: classifiers[i] };
        let label: String = mbti.to_string();
        let mut row: Vec<Value> = Vec::new();
        for (_j, feature) in sample.iter().enumerate() {
            let term: Value = Value::Number(serde_json::Number::from_f64(*feature).unwrap());
            row.push(term);
        }
        let idx = match label.as_str() {
            "ESFJ" => 0,
            "ESFP" => 1,
            "ESTJ" => 2,
            "ESTP" => 3,
            "ENFJ" => 4,
            "ENFP" => 5,
            "ENTJ" => 6,
            "ENTP" => 7,
            "ISFJ" => 8,
            "ISFP" => 9,
            "ISTJ" => 10,
            "ISTP" => 11,
            "INFJ" => 12,
            "INFP" => 13,
            "INTJ" => 14,
            "INTP" => 15,
            _ => panic!("Invalid label"),
        };
        data[idx].push(Value::Array(row));
    }
    for i in 0..16 {
        let label = match i {
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
        map.insert(label.to_string(), Value::Array(data[i].clone()));
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
    let (ie_corpus, ie_classifiers) = build_sets(
        &corpus,
        &classifiers,
        indicator::mb_flag::I,
        indicator::mb_flag::E,
    );
    let (ns_corpus, ns_classifiers) = build_sets(
        &corpus,
        &classifiers,
        indicator::mb_flag::N,
        indicator::mb_flag::S,
    );
    let (tf_corpus, tf_classifiers) = build_sets(
        &corpus,
        &classifiers,
        indicator::mb_flag::T,
        indicator::mb_flag::F,
    );
    let (jp_corpus, jp_classifiers) = build_sets(
        &corpus,
        &classifiers,
        indicator::mb_flag::J,
        indicator::mb_flag::P,
    );
    let ensemble = [
        (ie_corpus, ie_classifiers),
        (ns_corpus, ns_classifiers),
        (tf_corpus, tf_classifiers),
        (jp_corpus, jp_classifiers),
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
            train(&ensemble[i].0, &ensemble[i].1, &trees[i]);
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
    for (i, model) in models.iter().enumerate() {
        let x = DenseMatrix::from_2d_vec(&ensemble[i].0);
        let y = ensemble[i]
            .1
            .iter()
            .map(|x| *x as f64)
            .collect::<Vec<f64>>();
        let (_x_train, x_test, _y_train, y_test) = train_test_split(&x, &y, 0.2, true);
        ensemble_pred[i] = model.predict(&x_test).unwrap();
        println!(
            "{} accuracy: {}",
            trees[i],
            accuracy(&y_test, &ensemble_pred[i])
        );
        println!("MSE: {}", mean_squared_error(&y_test, &ensemble_pred[i]));
    }
    let mut svm_ensemble_y_pred: Vec<f64> = Vec::new();
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
            let variance = indicator_accuracy(&predicted, &actual);
            println!(
                "{},       {},   {:.2},     {:.2}           {}",
                predicted,
                actual,
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
        accuracy(&y_test, &svm_ensemble_y_pred)
    );
    println!("MSE: {}", mean_squared_error(&y_test, &svm_ensemble_y_pred));
    sample_report(&y_test, &svm_ensemble_y_pred);

    Ok(())
}
