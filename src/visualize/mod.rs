use std::path::{Path, PathBuf};

use image::ImageBuffer;
use nalgebra::DMatrix;
use rust_decimal::prelude::*;
use rust_ml_helpers::serialize::{load_bytes, write_file_truncate};
use serde_json::{Map, Value};

use crate::{
    csv_loader::{load_data, Sample},
    myers_briggs::{indicator, MBTI},
    normalize::load_processed_corpus,
    normalize::load_term_corpus,
    normalize::normalize,
};

const PATH_CLASSIFIERS_VALIDATED: &str = "./classifiers1.bincode";
const PATH_NORMALIZED_VISUAL_SIGNAL: &str = "./normalized_visual_signal.bincode";
const PATH_MBTI_SAMPLES_JSON: &str = "./mbti_samples.json";
const PATH_OVERALL_VISUAL: &str = "./overall_visual.bincode";

pub fn visualization_json() -> () {
    let path_classifiers1 = Path::new(PATH_CLASSIFIERS_VALIDATED);
    let data_y: Vec<u8> = bincode::deserialize(&load_bytes(path_classifiers1)).unwrap();
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
    let normalized_path = Path::new(PATH_NORMALIZED_VISUAL_SIGNAL);
    let visual_signal: Vec<Vec<f64>> = bincode::deserialize(&load_bytes(normalized_path)).unwrap();
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
    for (i, sample) in visual_signal.iter().enumerate() {
        let segment: DMatrix<f64> =
            DMatrix::from_fn(16, 16, |y, x| *sample.get((y * 16 + x) as usize).unwrap());
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
    let mbti_json_path = Path::new(PATH_MBTI_SAMPLES_JSON);
    if !mbti_json_path.exists() {
        let stringify = serde_json::to_string(&obj).unwrap();
        let bytes = stringify.as_bytes();
        write_file_truncate(mbti_json_path, &bytes.to_vec()).expect("Failed to write samples");
    }
}

pub fn create_charts() -> (Vec<Vec<f64>>, Vec<u8>) {
    let training_set: Vec<Sample> = load_data();
    let (tf_idf, classifiers) = normalize(training_set);
    let corpus = load_term_corpus();
    let (_dictionary, _df_index, overall_frequency, personality_frequency) =
        load_processed_corpus();
    let (overall, overall_terms) = {
        let path_overall = Path::new(PATH_OVERALL_VISUAL);
        let overall: DMatrix<f64>;
        if !path_overall.exists() {
            println!("loading required data.");
            println!("Creating overall visual signal...");
            overall = DMatrix::from_fn(corpus.nrows(), 16, |i, j| {
                if i + 1 % 25000 == 0 {
                    println!("overall: {} of {}", i, corpus.nrows());
                }
                corpus
                    .row(i)
                    .iter()
                    .enumerate()
                    .fold(0.0, |mut acc: f64, (n, term)| {
                        let mut val: f64 = 0.0;
                        if personality_frequency[j].contains_key(term) {
                            val = personality_frequency[j][term];
                        }
                        acc += ((val / overall_frequency[term])
                            * tf_idf.get(i).unwrap().get(n).unwrap())
                            as f64;
                        acc
                    })
                    / corpus.ncols() as f64
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
            write_file_truncate(path_overall, &bincode::serialize(&obj).unwrap())
                .expect("Failed to write overall");
            println!("Saved overall frequencies for each classification.");
        } else {
            println!("Loading overall frequencies for each classification...");
            let obj: Vec<Vec<f64>> = bincode::deserialize(&load_bytes(path_overall)).unwrap();
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
                    if personality_frequency[x].contains_key(term) {
                        val[x] = (personality_frequency[x][term] / overall_frequency[term])
                            * *tf_idf.get(i).unwrap().get(j).unwrap() as f64;
                    } else {
                        val[x] = 0.0;
                    }
                } else {
                    if personality_frequency[x].contains_key(term) {
                        val[x] = (personality_frequency[x][term] / overall_frequency[term]) as f64;
                    } else {
                        val[x] = 0.0;
                    }
                }
            }
            val
        });
        (overall, overall_terms)
    };

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
                                    [(i, (relative_y + 1) * (relative_x + 1) - 1)][*delta];
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
                        let norm_i = image8.normalize().slice((0, 0), (8, 4)).normalize();
                        let norm_e = image8.normalize().slice((0, 4), (8, 4)).normalize();
                        let norm_s = image8.normalize().slice((0, 8), (4, 8)).normalize();
                        let norm_n = image8.normalize().slice((4, 8), (4, 8)).normalize();
                        let norm_t = image8.normalize().slice((8, 0), (4, 8)).normalize();
                        let norm_f = image8.normalize().slice((12, 0), (4, 8)).normalize();
                        let norm_j = image8.normalize().slice((8, 8), (8, 4)).normalize();
                        let norm_p = image8.normalize().slice((8, 12), (8, 4)).normalize();
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
                                distance = ((yy - y1).powi(2) + (xx - x1).powi(2)).sqrt();
                            } else if angle == 45 {
                                let x1 = 8.0;
                                let y1 = 7.0;
                                distance = ((yy - y1).powi(2) + (xx - x1).powi(2)).sqrt();
                            } else if angle == 135 {
                                let x1 = 7.0;
                                let y1 = 8.0;
                                distance = ((yy - y1).powi(2) + (xx - x1).powi(2)).sqrt();
                            } else if angle == 225 {
                                let x1 = 8.0;
                                let y1 = 8.0;
                                distance = ((yy - y1).powi(2) + (xx - x1).powi(2)).sqrt();
                            }
                            distance
                        };
                        let height_at_x = |x, start, end: usize, x_start, x_end| -> (usize, f64) {
                            let initial_height = start as f64;
                            let final_height = end as f64;
                            let initial_x = x_start as f64;
                            let final_x = x_end as f64;
                            let slope = if WAVE {
                                (final_height - initial_height) / (final_x - initial_x)
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
                        } as usize;
                        let mut sn_height = if sum_s > sum_n {
                            8.0 + ((1.0 - ((sum_s / sum_s) * (sum_n / sum_s))) * 16.0)
                        } else {
                            7.0 - ((1.0 - ((sum_s / sum_n) * (sum_n / sum_n))) * 16.0)
                        } as usize;
                        let mut tf_height = if sum_t > sum_f {
                            8.0 + ((1.0 - ((sum_t / sum_t) * (sum_f / sum_t))) * 16.0)
                        } else {
                            7.0 - ((1.0 - ((sum_t / sum_f) * (sum_f / sum_f))) * 16.0)
                        } as usize;
                        let mut jp_height = if sum_j > sum_p {
                            8.0 + ((1.0 - ((sum_j / sum_j) * (sum_p / sum_j))) * 16.0)
                        } else {
                            7.0 - ((1.0 - ((sum_j / sum_p) * (sum_p / sum_p))) * 16.0)
                        } as usize;

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
                            let (ie_height_y, _ie_slope) =
                                height_at_x(x, ie_height, sn_height, ie_start, sn_start);
                            let (sn_height_y, _sn_slope) =
                                height_at_x(x, sn_height, tf_height, sn_start, tf_start);
                            let (tf_height_y, _tf_slope) =
                                height_at_x(x, tf_height, jp_height, tf_start, jp_start);
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
                            let i_height_y = radial_distance_xy(x, y, 315).round() as usize;
                            let e_height_y = radial_distance_xy(x, y, 315).round() as usize;
                            let s_height_y = radial_distance_xy(x, y, 45).round() as usize;
                            let n_height_y = radial_distance_xy(x, y, 45).round() as usize;
                            let t_height_y = radial_distance_xy(x, y, 135).round() as usize;
                            let f_height_y = radial_distance_xy(x, y, 135).round() as usize;
                            let j_height_y = radial_distance_xy(x, y, 225).round() as usize;
                            let p_height_y = radial_distance_xy(x, y, 225).round() as usize;
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
                                    || (y >= 8 && i_height_y.0 >= 8 && y <= i_height_y.0)
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
                                    || (y >= 8 && s_height_y.0 >= 8 && y <= s_height_y.0)
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
                                distance = ((yy - y1).powi(2) + (xx - x1).powi(2)).sqrt();
                            } else if angle == 45 {
                                let x1 = 8.0;
                                let y1 = 7.0;
                                distance = ((yy - y1).powi(2) + (xx - x1).powi(2)).sqrt();
                            } else if angle == 135 {
                                let x1 = 7.0;
                                let y1 = 8.0;
                                distance = ((yy - y1).powi(2) + (xx - x1).powi(2)).sqrt();
                            } else if angle == 225 {
                                let x1 = 8.0;
                                let y1 = 8.0;
                                distance = ((yy - y1).powi(2) + (xx - x1).powi(2)).sqrt();
                            }
                            distance
                        };
                        let height_at_x = |x, start, end: usize, x_start, x_end| -> (usize, f64) {
                            let initial_height = start as f64;
                            let final_height = end as f64;
                            let initial_x = x_start as f64;
                            let final_x = x_end as f64;
                            let slope = if WAVE {
                                (final_height - initial_height) / (final_x - initial_x)
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
                        } as usize;
                        let mut sn_height = if sum_s > sum_n {
                            8.0 + ((1.0 - ((sum_s / sum_s) * (sum_n / sum_s))) * 16.0)
                        } else {
                            7.0 - ((1.0 - ((sum_s / sum_n) * (sum_n / sum_n))) * 16.0)
                        } as usize;
                        let mut tf_height = if sum_t > sum_f {
                            8.0 + ((1.0 - ((sum_t / sum_t) * (sum_f / sum_t))) * 16.0)
                        } else {
                            7.0 - ((1.0 - ((sum_t / sum_f) * (sum_f / sum_f))) * 16.0)
                        } as usize;
                        let mut jp_height = if sum_j > sum_p {
                            8.0 + ((1.0 - ((sum_j / sum_j) * (sum_p / sum_j))) * 16.0)
                        } else {
                            7.0 - ((1.0 - ((sum_j / sum_p) * (sum_p / sum_p))) * 16.0)
                        } as usize;

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
                            let (ie_height_y, _ie_slope) =
                                height_at_x(x, ie_height, sn_height, ie_start, sn_start);
                            let (sn_height_y, _sn_slope) =
                                height_at_x(x, sn_height, tf_height, sn_start, tf_start);
                            let (tf_height_y, _tf_slope) =
                                height_at_x(x, tf_height, jp_height, tf_start, jp_start);
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
                            let i_height_y = radial_distance_xy(x, y, 315).round() as usize;
                            let e_height_y = radial_distance_xy(x, y, 315).round() as usize;
                            let s_height_y = radial_distance_xy(x, y, 45).round() as usize;
                            let n_height_y = radial_distance_xy(x, y, 45).round() as usize;
                            let t_height_y = radial_distance_xy(x, y, 135).round() as usize;
                            let f_height_y = radial_distance_xy(x, y, 135).round() as usize;
                            let j_height_y = radial_distance_xy(x, y, 225).round() as usize;
                            let p_height_y = radial_distance_xy(x, y, 225).round() as usize;
                            // This is where I tried to implement a polar area chart.
                            let i_y = if sum_i >= sum_e {
                                7.0 * sum_e / sum_i * sum_i / sum_i
                            } else {
                                7.0 * sum_i / sum_e * sum_e / sum_e
                            };
                            let e_y = if sum_i < sum_e {
                                7.0 * sum_i / sum_e * sum_e / sum_e
                            } else {
                                7.0 * sum_e / sum_i * sum_i / sum_i
                            };
                            let s_y = if sum_s >= sum_n {
                                8.0 * sum_n / sum_s * sum_s / sum_s
                            } else {
                                8.0 * sum_s / sum_n * sum_n / sum_n
                            };
                            let n_y = if sum_s < sum_n {
                                8.0 * sum_s / sum_n * sum_n / sum_n
                            } else {
                                8.0 * sum_n / sum_s * sum_s / sum_s
                            };
                            let t_y = if sum_t >= sum_f {
                                8.0 * sum_f / sum_t * sum_t / sum_t
                            } else {
                                8.0 * sum_t / sum_f * sum_f / sum_f
                            };
                            let f_y = if sum_t < sum_f {
                                8.0 * sum_t / sum_f * sum_f / sum_f
                            } else {
                                8.0 * sum_f / sum_t * sum_t / sum_t
                            };
                            let j_y = if sum_j >= sum_p {
                                7.0 * sum_p / sum_j * sum_j / sum_j
                            } else {
                                7.0 * sum_j / sum_p * sum_p / sum_p
                            };
                            let p_y = if sum_j < sum_p {
                                7.0 * sum_j / sum_p * sum_p / sum_p
                            } else {
                                7.0 * sum_p / sum_j * sum_j / sum_j
                            };
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
                                    || (y >= 8 && i_height_y.0 >= 8 && y <= i_height_y.0)
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
                                    || (y >= 8 && s_height_y.0 >= 8 && y <= s_height_y.0)
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
                if segment_normalized2[(7, 0)] == 1.0 && label & indicator::mb_flag::I == 0 {
                    valid = false;
                    invalid_reason.push_str("I");
                }
                // Check for Valid E.
                else if segment_normalized2[(0, 7)] == 1.0 && label & indicator::mb_flag::E == 0 {
                    valid = false;
                    invalid_reason.push_str("E");
                }
                // Check for Valid S.
                else if segment_normalized2[(0, 8)] == 1.0 && label & indicator::mb_flag::S == 0 {
                    valid = false;
                    invalid_reason.push_str("S");
                }
                // Check for Valid N.
                else if segment_normalized2[(7, 15)] == 1.0 && label & indicator::mb_flag::N == 0
                {
                    valid = false;
                    invalid_reason.push_str("N");
                }
                // Check for Valid T.
                else if segment_normalized2[(8, 0)] == 1.0 && label & indicator::mb_flag::T == 0 {
                    valid = false;
                    invalid_reason.push_str("T");
                }
                // Check for Valid F.
                else if segment_normalized2[(15, 7)] == 1.0 && label & indicator::mb_flag::F == 0
                {
                    valid = false;
                    invalid_reason.push_str("F");
                }
                // Check for Valid J.
                else if segment_normalized2[(15, 8)] == 1.0 && label & indicator::mb_flag::J == 0
                {
                    valid = false;
                    invalid_reason.push_str("J");
                }
                // Check for Valid P.
                else if segment_normalized2[(8, 15)] == 1.0 && label & indicator::mb_flag::P == 0
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
                            let decimal_val: Decimal = Decimal::from_f64(val / max).unwrap();
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
            write_file_truncate(path_heatmap, &bincode::serialize(&heatmap).unwrap())
                .expect("Failed to write heatmap");
            println!("Done.");
            println!("Saving classifiers1...");
            write_file_truncate(
                path_classifiers1,
                &bincode::serialize(&classifiers1).unwrap(),
            )
            .expect("Failed to write classifiers1");
            println!("Done.");
        } else {
            heatmap = bincode::deserialize(&load_bytes(path_heatmap)).unwrap();
            // classifiers1 = bincode::deserialize(&load_bytes(path_classifiers1)).unwrap();
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
    println!("Saving normalized visual signal...");
    let path_normalized = Path::new(PATH_NORMALIZED_VISUAL_SIGNAL);

    write_file_truncate(path_normalized, &bincode::serialize(&heatmap).unwrap())
        .expect("Failed to write visual signal");

    (heatmap, classifiers)
}
