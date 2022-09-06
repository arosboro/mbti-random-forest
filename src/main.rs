use std::fs::File;
use std::io::{Write, Read};
use std::path::Path;
use csv::Error;
use serde::{
  Serialize, 
  Deserialize
};
use smartcore::tree::decision_tree_classifier::SplitCriterion;
use std::collections::HashMap;
// DenseMatrix wrapper around Vec
use smartcore::linalg::naive::dense_matrix::*;
// Random Forest
use smartcore::ensemble::random_forest_classifier::{RandomForestClassifier, RandomForestClassifierParameters};
// Model performance
use smartcore::metrics::{mean_squared_error, accuracy};
use smartcore::model_selection::train_test_split;
use vtext::tokenize::*;
use rust_stemmers::{Algorithm, Stemmer};
use regex::Regex;
use std::collections::HashSet;
use stopwords::{Spark, Language, Stopwords};

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
type Dictionary = HashMap<String, f64>;

#[derive(Debug, Serialize, Deserialize)]
struct Sample {
  indicator: MBTI,
  posts: Vec<Vec<String>>,
}

fn cleanup(post: &str, expressions: &[Regex; 3]) -> String {
  let mut acc: String = post.to_owned();
  for expression in expressions.iter() {
    let rep = if expression.as_str() == r"\s+" { " " } else { "" };
    acc = expression.replace_all(&acc, rep).to_string();
  }
  acc.to_owned()
}

fn lemmatize(tokens: Post) -> Vec<String> {
  let mut lemmas = Vec::new();
  for token in tokens {
    let stemmer = Stemmer::create(Algorithm::English);
    let lemma = stemmer.stem(&token);
    lemmas.push(lemma.to_string());
  }
  lemmas
}

fn tokenize(post: &str, expressions: &[Regex; 3]) -> Vec<String> {
  let clean = cleanup(post.to_lowercase().as_str(), expressions);
  let stopswords: HashSet<_> = Spark::stopwords(Language::English).unwrap().iter().collect();
  let tokenizer = VTextTokenizerParams::default().lang("en").build().unwrap();
  let mut tokens: Vec<&str> = tokenizer.tokenize(&clean).collect();
  tokens.retain(|x| x.trim().len() > 0);
  tokens.retain(|token| !stopswords.contains(token));
  let clean_tokens = tokens.iter().map(|x| x.trim().to_string()).collect();
  lemmatize(clean_tokens)
}

fn load_data() -> Vec<Sample> {
  let samples: Vec<Sample> = {
    let path: &Path = Path::new("./samples.bincode");
    if path.exists() {
      println!("Loading samples...");
      let mut buf = Vec::new();
      File::open(path).unwrap()
        .read_to_end(&mut buf).expect("Unable to read file");
      let samples: Vec<Sample> = bincode::deserialize(&buf).unwrap();
      samples
    } else {
      println!("Saving samples...");
      let mut samples: Vec<Sample> = Vec::new();
      let mut reader = csv::Reader::from_path("./MBTI 500.csv").unwrap();
      let expressions = [
        Regex::new(r"https?://[a-zA-Z0-9_%./]+\??(([a-z0-9%]+=[a-z0-9%]+&?)+)?").unwrap(),
        Regex::new(r"[^a-zA-Z0-9 ]").unwrap(),
        Regex::new(r"\s+").unwrap(),
      ];
      for row in reader.deserialize::<Row>() {
        match row {
          Ok(row) => {
            let mut sample: Sample = Sample {
              indicator: MBTI::from_string(&row.r#type),
              posts: Vec::new(),
            };
            for post in row.posts.split("|||") {
              let tokens = tokenize(post, &expressions);
              if tokens.len() > 0 {
                sample.posts.push(tokens);
              }
            }
            samples.push(sample)
          },
          Err(e) => println!("Error: {}", e),
        }
      }
      let f = std::fs::OpenOptions::new().write(true).create(true).truncate(true).open(path);
      let samples_bytes = bincode::serialize(&samples).unwrap();
      f.and_then(|mut f| f.write_all(&samples_bytes)).expect("Failed to write samples");
      samples
    }
  };
  samples
}

fn normalize(training_set: &Vec<Sample>) -> (Vec<Vec<f64>>, Vec<u8>) {
  let path_fx = Path::new("./x_matrix.bincode");
  let path_fy = Path::new("./y_matrix.bincode");
  if !path_fx.exists() || !path_fy.exists() {
    println!("Saving x and y matrices...");
    let mut x_set: Vec<Vec<String>> = Vec::new();
    let mut y_set: Vec<MBTI> = Vec::new();
  
    for sample in training_set {
      for post in &sample.posts {
        x_set.push(post.to_owned());
        y_set.push(sample.indicator);
      }
    }
    println!("{} samples", x_set.len());
  
    // Deterimine unique labels
    let mut unique_labels: Vec<String> = Vec::new();
    for label in y_set.iter() {
      if !unique_labels.contains(&label.to_string()) {
        unique_labels.push(label.to_string());
      }
    }
    println!("{} unique labels", unique_labels.len());
  
    let dictionary: Dictionary = {
      let path: &Path = Path::new("./dictionary.bincode");
      if path.exists() {
        println!("Loading dictionary...");
        let mut buf = Vec::new();
        File::open(path).unwrap()
          .read_to_end(&mut buf).expect("Unable to read file");
        let dictionary: Dictionary = bincode::deserialize(&buf).unwrap();
        dictionary
      }
      else {
        println!("Saving dictionary...");
        // Create a dictionary indexing unique tokens.
        let mut dictionary: Dictionary = HashMap::new();
        for post in x_set.clone() {
          for token in post {
            if !dictionary.contains_key(&token.to_string()) {
              dictionary.insert(token.to_string(), dictionary.len() as f64);
            }
          }
        }
        let f = std::fs::OpenOptions::new().write(true).create(true).truncate(true).open(path);
        let dictionary_bytes = bincode::serialize(&dictionary).unwrap();
        f.and_then(|mut f| f.write_all(&dictionary_bytes)).expect("Failed to write dictionary");
        dictionary
      }
    };
  
    println!("Dictionary size: {}", dictionary.len());
  
    // Create f64 matrices for x_set.
    let mut x_matrix: Vec<Vec<f64>> = Vec::new();
    for post in x_set {
      let mut matrix: Vec<f64> = Vec::new();
      for token in post {
        let index: f64 = dictionary.get_key_value(&token.to_string()).unwrap().1.clone();
        matrix.push(index);
      }
      x_matrix.push(matrix);
    }
  
    // Create f64 matrices for y_set.
    let mut y_matrix: Vec<u8> = Vec::new();
    for indicator in y_set {
      let index: u8 = indicator.indicator;
      y_matrix.push(index);
    }

    let x_matrix_bytes = bincode::serialize(&x_matrix).expect("Can not serialize the matrix");
          File::create("x_matrix.bincode")
            .and_then(|mut f| f.write_all(&x_matrix_bytes))
            .expect("Can not persist x_matrix");
    let y_matrix_bytes: Vec<u8> = bincode::serialize(&y_matrix).expect("Can not serialize the matrix");
          File::create("y_matrix.bincode")
            .and_then(|mut f| f.write_all(&y_matrix_bytes))
            .expect("Can not persist y_matrix");
    (x_matrix, y_matrix)
  }
  else {
    println!("Loading x and y matrices...");
    let mut x_buf = Vec::new();
    let mut y_buf = Vec::new();
    std::fs::File::open("x_matrix.bincode").expect("Can not open x_matrix").read_to_end(&mut x_buf).unwrap();
    std::fs::File::open("y_matrix.bincode").expect("Can not open y_matrix").read_to_end(&mut y_buf).unwrap();
    let x_matrix: Vec<Vec<f64>> = bincode::deserialize(&x_buf).unwrap();
    let y_matrix: Vec<u8> = bincode::deserialize(&y_buf).unwrap();
    (x_matrix, y_matrix)
  }
}

fn tally(y_matrix: &Vec<u8>) {
  println!("Tallying...");
  println!("count ISTJ: {}", y_matrix.iter().filter(|&m| *m == indicator::ISTJ).count());
  println!("count ISFJ: {}", y_matrix.iter().filter(|&m| *m == indicator::ISFJ).count());
  println!("count INFJ: {}", y_matrix.iter().filter(|&m| *m == indicator::INFJ).count());
  println!("count INTJ: {}", y_matrix.iter().filter(|&m| *m == indicator::INTJ).count());
  println!("count ISTP: {}", y_matrix.iter().filter(|&m| *m == indicator::ISTP).count());
  println!("count ISFP: {}", y_matrix.iter().filter(|&m| *m == indicator::ISFP).count());
  println!("count INFP: {}", y_matrix.iter().filter(|&m| *m == indicator::INFP).count());
  println!("count INTP: {}", y_matrix.iter().filter(|&m| *m == indicator::INTP).count());
  println!("count ESTP: {}", y_matrix.iter().filter(|&m| *m == indicator::ESTP).count());
  println!("count ESFP: {}", y_matrix.iter().filter(|&m| *m == indicator::ESFP).count());
  println!("count ENFP: {}", y_matrix.iter().filter(|&m| *m == indicator::ENFP).count());
  println!("count ENTP: {}", y_matrix.iter().filter(|&m| *m == indicator::ENTP).count());
  println!("count ESTJ: {}", y_matrix.iter().filter(|&m| *m == indicator::ESTJ).count());
  println!("count ESFJ: {}", y_matrix.iter().filter(|&m| *m == indicator::ESFJ).count());
  println!("count ENFJ: {}", y_matrix.iter().filter(|&m| *m == indicator::ENFJ).count());
  println!("count ENTJ: {}", y_matrix.iter().filter(|&m| *m == indicator::ENTJ).count());
}


fn train(x_matrix: &Vec<Vec<f64>>, y_matrix: &Vec<u8>, member_id: &str) {
  println!("Training...");
  let x: DenseMatrix<f64> = DenseMatrix::from_2d_vec(&x_matrix);
  // These are our target class labels
  let y: Vec<f64> = y_matrix.into_iter().map(|x| *x as f64).collect();
  // Split bag into training/test (80%/20%)
  let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true);

  // Parameters
  const DEFAULT_PARAMS: RandomForestClassifierParameters = RandomForestClassifierParameters {
    criterion: SplitCriterion::Gini,
    max_depth: None,
    min_samples_leaf: 1,
    min_samples_split: 2,
    n_trees: 100,
    m: Option::None,
    keep_samples: false,
    seed: 0,
  };

  println!("{:?}", DEFAULT_PARAMS);

  const TWEAKED_PARAMS: RandomForestClassifierParameters = RandomForestClassifierParameters {
    /// Split criteria to use when building a tree. See [Decision Tree Classifier](../../tree/decision_tree_classifier/index.html)
    criterion: SplitCriterion::Gini,
    /// Tree max depth. See [Decision Tree Classifier](../../tree/decision_tree_classifier/index.html)
    max_depth: Some(8),
    /// The minimum number of samples required to be at a leaf node. See [Decision Tree Classifier](../../tree/decision_tree_classifier/index.html)
    min_samples_leaf: 128,
    /// The minimum number of samples required to split an internal node. See [Decision Tree Classifier](../../tree/decision_tree_classifier/index.html)
    min_samples_split: 256,
    /// The number of trees in the forest.
    n_trees: 16,
    /// Number of random sample of predictors to use as split candidates.
    m: Some(64),
    /// Whether to keep samples used for tree generation. This is required for OOB prediction.
    keep_samples: true,
    /// Seed used for bootstrap sampling and feature selection for each tree.
    seed: 42u64,
  };

  println!("{:?}", TWEAKED_PARAMS);

  // Random Forest
  for (iteration, rf) in RandomForestClassifier::fit(&x_train, &y_train, TWEAKED_PARAMS).iter().enumerate() {
    println!("Serializing random forest...");
    let bytes_rf = bincode::serialize(&rf).unwrap();
    File::create(format!("mbti_rf__{}.model", member_id))
      .and_then(|mut f| f.write_all(&bytes_rf))
      .expect(format!("Can not persist random_forest {}", member_id).as_str());
    let y_pred: Vec<f64> = rf.predict(&x_test).unwrap();
    println!("Iteration: {}", iteration);
    println!("Random Forest accuracy: {}", accuracy(&y_test, &y_pred));
    println!("MSE: {}", mean_squared_error(&y_test, &y_pred));
  }
  
  // Load the Model
  println!("Loading random forest...");
  let rf: RandomForestClassifier<f64> = {
    let mut buf: Vec<u8> = Vec::new();
    File::open(format!("mbti_rf__{}.model", member_id))
      .and_then(|mut f| f.read_to_end(&mut buf))
      .expect("Can not load model");
    bincode::deserialize(&buf).expect("Can not deserialize the model")
  };
  

  println!("Validation of serialized model...");
  let y_pred = rf.predict(&x_test).unwrap();
  // assert!(y_hat_rf == y_pred, "Inconsistent models.");

  // Calculate the accuracy
  println!("Metrics about model.");
  println!("Accuracy: {}", accuracy(&y_test, &y_pred));
  // Calculate test error
  println!("MSE: {}", mean_squared_error(&y_test, &y_pred));
}

fn build_sets(x_matrix: &Vec<Vec<f64>>, y_matrix: &Vec<u8>, leaf_a: u8, leaf_b: u8) -> (Vec<Vec<f64>>, Vec<u8>) {
  let mut x_matrix_set: Vec<Vec<f64>> = Vec::new();
  let mut y_matrix_set: Vec<u8> = Vec::new();
  for (i, y) in y_matrix.iter().enumerate() {
    let left = *y & leaf_a != 0u8;
    let right = *y & leaf_b != 0u8;
    if left {
      x_matrix_set.push(x_matrix[i].clone());
      y_matrix_set.push(0u8);
    } else if right {
      x_matrix_set.push(x_matrix[i].clone());
      y_matrix_set.push(1u8);
    }
    else {
      continue;
    }
  }
  (x_matrix_set, y_matrix_set)
}

fn main() -> Result<(), Error> {
  let training_set: Vec<Sample> = load_data();
  let (x_matrix, y_matrix) = normalize(&training_set);
  tally(&y_matrix);
  // Build sets for an ensemble of models
  let (ie_x_matrix, ie_y_matrix) = build_sets(&x_matrix, &y_matrix, indicator::mb_flag::I, indicator::mb_flag::E);
  let (ns_x_matrix, ns_y_matrix) = build_sets(&x_matrix, &y_matrix, indicator::mb_flag::N, indicator::mb_flag::S);
  let (tf_x_matrix, tf_y_matrix) = build_sets(&x_matrix, &y_matrix, indicator::mb_flag::T, indicator::mb_flag::F);
  let (jp_x_matrix, jp_y_matrix) = build_sets(&x_matrix, &y_matrix, indicator::mb_flag::J, indicator::mb_flag::P);
  let ensemble = [(ie_x_matrix, ie_y_matrix), (ns_x_matrix, ns_y_matrix), (tf_x_matrix, tf_y_matrix), (jp_x_matrix, jp_y_matrix)];
  // Train models
  let tree: [&str; 4] = ["IE", "NS", "TF", "JP"];
  for i in 0..4 {
    println!{"Tally of [IE, NS, TF, JP]: {}", tree[i]};
    println!{"{} samples for {}", ensemble[i].1.iter().filter(|&n| *n == 0u8).count(), tree[i].chars().nth(0).unwrap()};
    println!{"{} samples for {}", ensemble[i].1.iter().filter(|&n| *n == 1u8).count(), tree[i].chars().nth(1).unwrap()};
  }

  if !Path::new("./mbti_rf__ALL.model").exists() {
    println!("Generating generic model");
    train(&x_matrix, &y_matrix, "ALL");
  } else {
    println!("Generic model already exists");
    // TODO load model and test
  }

  let mut models: Vec<RandomForestClassifier<f64>> = Vec::new();
  for i in 0..4 {
    let filename = format!("./mbti_rf__{}.model", tree[i]);
    println!{"Training [IE, NS, TF, JP]: {}", tree[i]};
    let path_model = Path::new(&filename);
    if !path_model.exists() {
      train(&ensemble[i].0, &ensemble[i].1, tree[i]);
    }
    else {
      println!("Loading random forest {} model...", tree[i]);
      let rf: RandomForestClassifier<f64> = {
        let mut buf: Vec<u8> = Vec::new();
        File::open(path_model)
          .and_then(|mut f| f.read_to_end(&mut buf))
          .expect("Can not load model");
        bincode::deserialize(&buf).expect("Can not deserialize the model")
      };
      models.push(rf);
    }
  }

  let x: DenseMatrix<f64> = DenseMatrix::from_2d_vec(&x_matrix);
  // These are our target class labels
  let y: Vec<f64> = y_matrix.into_iter().map(|x| x as f64).collect();
  // Split bag into training/test (80%/20%)
  let (_x_train, x_test, _y_train, y_test) = train_test_split(&x, &y, 0.2, true);
  let mut ensemble_pred: [Vec<f64>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
  for (i, model) in models.iter().enumerate() {
    ensemble_pred[i] = model.predict(&x_test).unwrap();
  }
  let mut y_pred: Vec<f64> = Vec::new();
  for i in 0..y_test.len(){
    let mut mbti: u8 = 0u8;
    for j in 0..4 {
      if ensemble_pred[j][i] == 0f64 && tree[j].chars().nth(0).unwrap() == 'I' {
        mbti ^= indicator::mb_flag::I;
      }
      else if ensemble_pred[j][i] == 1f64 && tree[j].chars().nth(1).unwrap() == 'E' {
        mbti ^= indicator::mb_flag::E;
      }
      else if ensemble_pred[j][i] == 0f64 && tree[j].chars().nth(0).unwrap() == 'N' {
        mbti ^= indicator::mb_flag::N;
      }
      else if ensemble_pred[j][i] == 1f64 && tree[j].chars().nth(1).unwrap() == 'S' {
        mbti ^= indicator::mb_flag::S;
      }
      else if ensemble_pred[j][i] == 0f64 && tree[j].chars().nth(0).unwrap() == 'T' {
        mbti ^= indicator::mb_flag::T;
      }
      else if ensemble_pred[j][i] == 1f64 && tree[j].chars().nth(1).unwrap() == 'F' {
        mbti ^= indicator::mb_flag::F;
      }
      else if ensemble_pred[j][i] == 0f64 && tree[j].chars().nth(0).unwrap() == 'J' {
        mbti ^= indicator::mb_flag::J;
      }
      else if ensemble_pred[j][i] == 1f64 && tree[j].chars().nth(1).unwrap() == 'P' {
        mbti ^= indicator::mb_flag::P;
      }
    }
    y_pred.push(mbti as f64);
  }
  // Calculate the accuracy
  println!("Metrics about model.");
  println!("Accuracy: {}", accuracy(&y_test, &y_pred));
  // Calculate test error
  println!("MSE: {}", mean_squared_error(&y_test, &y_pred));

  Ok(()) 
}
