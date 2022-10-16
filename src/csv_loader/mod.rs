use rand::seq::SliceRandom;
use rand::thread_rng;
use regex::Regex;
use rust_ml_helpers::preprocess::*;
use rust_ml_helpers::serialize::load_bytes;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::Write;
use std::path::Path;

use crate::myers_briggs::MBTI;

#[derive(Debug, Deserialize)]
struct Row {
    r#type: String,
    posts: String,
}

pub type Lemma = String;
pub type Lemmas = Vec<Lemma>;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Sample {
    pub label: MBTI,
    pub lemmas: Lemmas,
}

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

fn get_stopwords() -> HashSet<&'static str> {
    let mut stopwords: Vec<&'static str> = Vec::new();
    stopwords.extend(ENNEAGRAM_TERMS.iter());
    stopwords.extend(IMAGE_TERMS.iter());
    stopwords.extend(INVALID_WORDS.iter());
    stopwords.extend(MBTI_TERMS.iter());
    stopwords.extend(MBTI_PARTS.iter());
    stopwords::get(stopwords)
}

pub fn load_data() -> Vec<Sample> {
    const BALANCED_SAMPLES: bool = true;
    const MANUAL_VS_AVG: bool = true;
    const MANUAL_POST_SIZE: usize = 32;
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
            let expressions: [Regex; 3] = [
                Regex::new(r"https?://[a-zA-Z0-9_%./]+\??(([a-z0-9%]+=[a-z0-9%]+&?)+)?").unwrap(),
                Regex::new(r"[^a-zA-Z0-9 ]").unwrap(),
                Regex::new(r"\s+").unwrap(),
            ];
            let mut counters: HashMap<u8, usize> = HashMap::new();
            for row in reader.deserialize::<Row>() {
                match row {
                    Ok(row) => {
                        // Choose the first MANUAL_POST_SIZE lemmas for a composite post.
                        if MANUAL_VS_AVG {
                            let mut lemma_group: Vec<String> = Vec::new();
                            for post in row.posts.split("|||").collect::<Vec<&str>>() {
                                let lemmas = tokenize(post, expressions.to_vec(), get_stopwords());
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
                        // Use the average length of all posts for rejecting a post from the set of Samples.
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
                                        lemmas: tokenize(
                                            post,
                                            expressions.to_vec(),
                                            get_stopwords(),
                                        ),
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
            f.and_then(|mut f: File| f.write_all(&samples_bytes))
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
