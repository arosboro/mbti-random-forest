use std::fs::File;
use std::io::{Write, Read};
use std::path::Path;
use std::time::Instant;
use csv::Error;
use serde::{
  Serialize, 
  Deserialize
};
use smartcore::tree::decision_tree_classifier::SplitCriterion;
use std::collections::{HashMap, HashSet};
// DenseMatrix wrapper around Vec
use smartcore::linalg::naive::dense_matrix::*;
// Random Forest
use smartcore::ensemble::random_forest_classifier::{RandomForestClassifier, RandomForestClassifierParameters};
use smartcore::svm::svc::{SVCParameters, SVC};
use smartcore::metrics::roc_auc_score;
// Model performance
use smartcore::metrics::{mean_squared_error, accuracy};
use smartcore::model_selection::train_test_split;
use vtext::tokenize::*;
use rust_stemmers::{Algorithm, Stemmer};
use regex::Regex;
use stopwords::{Spark, Language, Stopwords};
use nalgebra::{DMatrix, DMatrixSlice};


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
  let stopwords: HashSet<_> = Spark::stopwords(Language::English).unwrap().iter().collect();
  let tokenizer = VTextTokenizerParams::default().lang("en").build().unwrap();
  let mut tokens: Vec<&str> = tokenizer.tokenize(&clean).collect();
  tokens.retain(|x| x.trim().len() > 0);
  tokens.retain(|token| !stopwords.contains(token));
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
              let mut post_vec = Vec::new();
              if tokens.len() > 0 {
                tokens.iter().enumerate().for_each(|(i, _)| {
                  post_vec.push(tokens[i].to_owned());
                });
                sample.posts.push(post_vec);
              }
            }
            samples.push(sample)
          },
          Err(e) => println!("Error: {}", e),
        }
      }
      let mut count_row = samples[0].posts[0].len();
      for sample in &samples {
        for post in &sample.posts {
          if post.len() < count_row && post.len() > 0 {
            count_row = post.len();
          }
        }
      }
      let mut samples_truncated: Vec<Sample>  = Vec::new();
      for sample in &samples {
        let mut truncated_sample: Sample = Sample {
          indicator: sample.indicator,
          posts: Vec::new(),
        };
        for post in &sample.posts {
          let mut post_truncated: Vec<String> = Vec::new();
          if post.len() > 0 {
            for i in 0..count_row {
              post_truncated.push(post[i].to_owned());
            }
            truncated_sample.posts.push(post_truncated);
          }
        }
        samples_truncated.push(truncated_sample);
      };
      let f = std::fs::OpenOptions::new().write(true).create(true).truncate(true).open(path);
      let samples_bytes = bincode::serialize(&samples_truncated).unwrap();
      f.and_then(|mut f| f.write_all(&samples_bytes)).expect("Failed to write samples");
      samples_truncated
    }
  };
  samples
}

fn normalize(training_set: &Vec<Sample>) -> (Vec<Vec<f64>>, Vec<u8>) {
  let path_fx = Path::new("./corpus.bincode");
  let path_fy = Path::new("./classifiers.bincode");
  if !path_fx.exists() || !path_fy.exists() {
    println!("Saving x and y matrices...");
    let mut x_set: Vec<Vec<String>> = Vec::new();
    for sample in training_set.iter() {
      for post in sample.posts.iter() {
        x_set.push(post.to_owned());
      }
    };
    let y_set: Vec<u8> = training_set.iter().map(|x| x.indicator.indicator).collect();
    println!("{} x samples", x_set.len());
    println!("{} y labels", y_set.len());

    let corpus: DMatrix<String> = DMatrix::from_fn(x_set.len(), x_set[0].len(), |i, j| x_set[i][j].to_owned());
    let classifiers: DMatrix<u8> = DMatrix::from_fn(y_set.len(), 1, |i, _| y_set[i]);
   
    // Deterimine unique labels
    let mut unique_labels: Vec<String> = Vec::new();
    for label in y_set.iter() {
      let mbti = MBTI{ indicator: *label };
      if !unique_labels.contains(&mbti.to_string()) {
        unique_labels.push(mbti.to_string());
      }
    }
    println!("{} unique labels", unique_labels.len());
  
    let (dictionary, df_index) = {
      let path_dict: &Path = Path::new("./dictionary.bincode");
      let path_df: &Path = Path::new("./df_index.bincode");
      if path_dict.exists() && path_df.exists() {
        println!("Loading dictionary...");
        let mut buf = Vec::new();
        File::open(path_dict).unwrap()
          .read_to_end(&mut buf).expect("Unable to read file");
        let dictionary: Dictionary = bincode::deserialize(&buf).unwrap();
        println!("Loading df_index...");
        buf = Vec::new();
        File::open(path_df).unwrap()
          .read_to_end(&mut buf).expect("Unable to read file");
        let df_index: Dictionary = bincode::deserialize(&buf).unwrap();
        (dictionary, df_index)
      }
      else {
        println!("Saving dictionary and df_index...");
        // Create a dictionary indexing unique tokens.
        // Also create a df_index containing the number of documents each token appears in.
        let mut dictionary: Dictionary = HashMap::new();
        let mut df_major: HashMap<String, f64> = HashMap::new();
        for post in x_set {
          let df_minor: HashMap<String, f64> = post.iter().fold(HashMap::new(), |mut acc, token| {
            *acc.entry(token.to_owned()).or_insert(0.0) += 1.0;
            acc
          });
          for token in post {
            if !dictionary.contains_key(&token.to_string()) {
              dictionary.insert(token.to_string(), dictionary.len() as f64);
            }
          }
          df_minor.iter().for_each(|(token, _)| {
            *df_major.entry(token.to_string()).or_insert(0.0) += 1.0;
          });
        }
        // Serialize the dictionary.
        println!("Saving dictionary...");
        let mut f = std::fs::OpenOptions::new().write(true).create(true).truncate(true).open(path_dict);
        let dictionary_bytes = bincode::serialize(&dictionary).unwrap();
        f.and_then(|mut f| f.write_all(&dictionary_bytes)).expect("Failed to write dictionary");
        // Serialize the df_index.
        println!("Saving df_index...");
        f = std::fs::OpenOptions::new().write(true).create(true).truncate(true).open(path_df);
        let df_index_bytes = bincode::serialize(&df_major).unwrap();
        f.and_then(|mut f| f.write_all(&df_index_bytes)).expect("Failed to write df_index");
        (dictionary, df_major)
      }
    };

    println!("Dictionary size: {}", dictionary.len());

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
        }
        else {
          0.0
        }
      }).sum()
    };
    let idf = |term: &String| -> f64 {
      // Smooth inverse formula by adding 1.0 to denominator to prevent division by zero
      (corpus.nrows() as f64 / (df_index[term] + 1.0)).ln() as f64
    };

    // Create a dense matrix of term frequencies.
    let tf_matrix: Vec<Vec<f64>> = {
      let path = Path::new("./tf_matrix.bincode");
      if path.exists() {
        println!("Loading tf_matrix...");
        let mut buf = Vec::new();
        File::open(path).unwrap()
          .read_to_end(&mut buf).expect("Unable to read file");
        let tf_matrix: Vec<Vec<f64>> = bincode::deserialize(&buf).unwrap();
        tf_matrix
      }
      else {
        println!("Creating a dense matrix of term frequencies...");
        let start = Instant::now();
        let tf_matrix: DMatrix<f64> = DMatrix::from_fn(corpus.nrows(), corpus.ncols(), |i, j| -> f64 {
          tf(corpus.slice((i, 0), (1, corpus.ncols())), &corpus[(i, j)])
        });
        println!("tf_matrix: {} minutes", start.elapsed().as_secs() / 60);
        println!("We obtained a {}x{} matrix", tf_matrix.nrows(), tf_matrix.ncols());
        let mut corpus_tf: Vec<Vec<f64>> = Vec::new();
        for i in 0..tf_matrix.nrows() {
          let mut row: Vec<f64> = Vec::new();
          for j in 0..tf_matrix.ncols() {
            row.push(tf_matrix[(i, j)]);
          }
          corpus_tf.push(row);
        }
        println!("Saving tf_matrix...");
        let f = std::fs::OpenOptions::new().write(true).create(true).truncate(true).open(path);
        let tf_matrix_bytes = bincode::serialize(&corpus_tf).unwrap();
        f.and_then(|mut f| f.write_all(&tf_matrix_bytes)).expect("Failed to write tf_matrix");
        corpus_tf
      }
    };
    

    // Create a dense matrix of idf values.
    let idf_matrix: Vec<Vec<f64>> = {
      let path = Path::new("./idf_matrix.bincode");
      if path.exists() {
        println!("Loading idf_matrix...");
        let mut buf = Vec::new();
        File::open(path).unwrap()
          .read_to_end(&mut buf).expect("Unable to read file");
        let idf_matrix: Vec<Vec<f64>> = bincode::deserialize(&buf).unwrap();
        idf_matrix
      } else {
        println!("Creating a dense matrix of idf values...");
        let start = Instant::now();
        let idf_matrix: DMatrix<f64> = DMatrix::from_fn(corpus.nrows(), corpus.ncols(), |i, j| -> f64 {
          idf(&corpus[(i, j)])
        });
        println!("idf_matrix: {} minutes", start.elapsed().as_secs() / 60);
        println!("We obtained a {}x{} matrix", idf_matrix.nrows(), idf_matrix.ncols());
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
        let f = std::fs::OpenOptions::new().write(true).create(true).truncate(true).open(path);
        let idf_matrix_bytes = bincode::serialize(&corpus_idf).unwrap();
        f.and_then(|mut f| f.write_all(&idf_matrix_bytes)).expect("Failed to write idf_matrix");
        corpus_idf
      }
    };

    // Finally, create the tf-idf matrix by multiplying.
    let tf_idf: Vec<Vec<f64>> = {
      let path = Path::new("./tf_idf.bincode");
      if path.exists() {
        println!("Loading tf_idf...");
        let mut buf = Vec::new();
        File::open(path).unwrap()
          .read_to_end(&mut buf).expect("Unable to read file");
        let tf_idf: Vec<Vec<f64>> = bincode::deserialize(&buf).unwrap();
        tf_idf
      } else {
        println!("Creating the tf-idf matrix by multiplying...");
        let start = Instant::now();
        let tf_idf: DMatrix<f64> = DMatrix::from_fn(corpus.nrows(), corpus.ncols(), |i, j| -> f64 {
          tf_matrix[i][j] * idf_matrix[i][j]
        }).normalize();
        println!("tf_idf: {} minutes", start.elapsed().as_secs() / 60);
        // Convert tf_idf to a Vec<Vec<f64>>.
        let mut corpus_tf_idf: Vec<Vec<f64>> = Vec::new();
        for i in 0..tf_idf.nrows() {
          let mut row: Vec<f64> = Vec::new();
          for j in 0..tf_idf.ncols() {
            row.push(tf_idf[(i, j)]);
          }
          corpus_tf_idf.push(row);
        }
        println!("Saving tf_idf...");
        let f = std::fs::OpenOptions::new().write(true).create(true).truncate(true).open(path);
        let tf_idf_bytes = bincode::serialize(&corpus_tf_idf).unwrap();
        f.and_then(|mut f| f.write_all(&tf_idf_bytes)).expect("Failed to write tf_idf");
        corpus_tf_idf
      }
    };

    // Create f64 matrices for y_set.
    let classifiers: Vec<u8> = classifiers.iter().map(|y| *y).collect();

    let corpus_bytes = bincode::serialize(&tf_idf).expect("Can not serialize the matrix");
          File::create("corpus.bincode")
            .and_then(|mut f| f.write_all(&corpus_bytes))
            .expect("Can not persist corpus");
    let classifiers_bytes: Vec<u8> = bincode::serialize(&classifiers).expect("Can not serialize the matrix");
          File::create("classifiers.bincode")
            .and_then(|mut f| f.write_all(&classifiers_bytes))
            .expect("Can not persist classifiers");
    (tf_idf, classifiers)
  } else {
    println!("Loading x and y matrices...");
    let mut x_buf = Vec::new();
    let mut y_buf = Vec::new();
    std::fs::File::open("corpus.bincode").expect("Can not open corpus").read_to_end(&mut x_buf).unwrap();
    std::fs::File::open("classifiers.bincode").expect("Can not open classifiers").read_to_end(&mut y_buf).unwrap();
    let corpus: Vec<Vec<f64>> = bincode::deserialize(&x_buf).unwrap();
    let classifiers: Vec<u8> = bincode::deserialize(&y_buf).unwrap();
    (corpus, classifiers)
  }
}

fn tally(classifiers: &Vec<u8>) {
  println!("Tallying...");
  println!("count ISTJ: {}", classifiers.iter().filter(|&m| *m == indicator::ISTJ).count());
  println!("count ISFJ: {}", classifiers.iter().filter(|&m| *m == indicator::ISFJ).count());
  println!("count INFJ: {}", classifiers.iter().filter(|&m| *m == indicator::INFJ).count());
  println!("count INTJ: {}", classifiers.iter().filter(|&m| *m == indicator::INTJ).count());
  println!("count ISTP: {}", classifiers.iter().filter(|&m| *m == indicator::ISTP).count());
  println!("count ISFP: {}", classifiers.iter().filter(|&m| *m == indicator::ISFP).count());
  println!("count INFP: {}", classifiers.iter().filter(|&m| *m == indicator::INFP).count());
  println!("count INTP: {}", classifiers.iter().filter(|&m| *m == indicator::INTP).count());
  println!("count ESTP: {}", classifiers.iter().filter(|&m| *m == indicator::ESTP).count());
  println!("count ESFP: {}", classifiers.iter().filter(|&m| *m == indicator::ESFP).count());
  println!("count ENFP: {}", classifiers.iter().filter(|&m| *m == indicator::ENFP).count());
  println!("count ENTP: {}", classifiers.iter().filter(|&m| *m == indicator::ENTP).count());
  println!("count ESTJ: {}", classifiers.iter().filter(|&m| *m == indicator::ESTJ).count());
  println!("count ESFJ: {}", classifiers.iter().filter(|&m| *m == indicator::ESFJ).count());
  println!("count ENFJ: {}", classifiers.iter().filter(|&m| *m == indicator::ENFJ).count());
  println!("count ENTJ: {}", classifiers.iter().filter(|&m| *m == indicator::ENTJ).count());
}


fn train(corpus: &Vec<Vec<f64>>, classifiers: &Vec<u8>, member_id: &str) {
  println!("Training...");
  let x: DenseMatrix<f64> = DenseMatrix::from_2d_vec(&corpus);
  // These are our target class labels
  let y: Vec<f64> = classifiers.into_iter().map(|x| *x as f64).collect();
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
  let y_hat_rf: Vec<f64> = RandomForestClassifier::fit(&x_train, &y_train, TWEAKED_PARAMS)
    .and_then(|rf| {
      println!("Serializing random forest...");
      let bytes_rf = bincode::serialize(&rf).unwrap();
      File::create(format!("mbti_rf__{}.model", member_id))
        .and_then(|mut f| f.write_all(&bytes_rf))
        .expect(format!("Can not persist random_forest {}", member_id).as_str());
      rf.predict(&x_test)
    })
    .unwrap();
  println!("Random Forest accuracy: {}", accuracy(&y_test, &y_hat_rf));
  println!("MSE: {}", mean_squared_error(&y_test, &y_hat_rf));

  // SVM
  let y_hat_svm: Vec<f64> = SVC::fit(&x_train, &y_train, SVCParameters::default().with_c(200.0))
    .and_then(|svm| {
      let bytes_rf = bincode::serialize(&svm).unwrap();
      File::create(format!("mbti_svm__{}.model", member_id))
        .and_then(|mut f| f.write_all(&bytes_rf))
        .expect(format!("Can not persist svm {}", member_id).as_str());
      svm.predict(&x_test)
    })
    .unwrap();
  // Calculate test error    
  println!("Random Forest accuracy: {}", accuracy(&y_test, &y_hat_rf));
  println!("MSE: {}", mean_squared_error(&y_test, &y_hat_rf));
  println!("AUC SVM: {}", roc_auc_score(&y_test, &y_hat_svm));
  
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

fn build_sets(corpus: &Vec<Vec<f64>>, classifiers: &Vec<u8>, leaf_a: u8, leaf_b: u8) -> (Vec<Vec<f64>>, Vec<u8>) {
  let mut corpus_set: Vec<Vec<f64>> = Vec::new();
  let mut classifiers_set: Vec<u8> = Vec::new();
  for (i, y) in classifiers.iter().enumerate() {
    let left = *y & leaf_a != 0u8;
    let right = *y & leaf_b != 0u8;
    if left {
      corpus_set.push(corpus[i].clone());
      classifiers_set.push(0u8);
    } else if right {
      corpus_set.push(corpus[i].clone());
      classifiers_set.push(1u8);
    }
    else {
      continue;
    }
  }
  (corpus_set, classifiers_set)
}

fn main() -> Result<(), Error> {
  let training_set: Vec<Sample> = load_data();
  let (corpus, classifiers) = normalize(&training_set);
  tally(&classifiers);
  // Build sets for an ensemble of models
  let (ie_corpus, ie_classifiers) = build_sets(&corpus, &classifiers, indicator::mb_flag::I, indicator::mb_flag::E);
  let (ns_corpus, ns_classifiers) = build_sets(&corpus, &classifiers, indicator::mb_flag::N, indicator::mb_flag::S);
  let (tf_corpus, tf_classifiers) = build_sets(&corpus, &classifiers, indicator::mb_flag::T, indicator::mb_flag::F);
  let (jp_corpus, jp_classifiers) = build_sets(&corpus, &classifiers, indicator::mb_flag::J, indicator::mb_flag::P);
  let ensemble = [(ie_corpus, ie_classifiers), (ns_corpus, ns_classifiers), (tf_corpus, tf_classifiers), (jp_corpus, jp_classifiers)];
  // Train models
  let tree: [&str; 4] = ["IE", "NS", "TF", "JP"];
  for i in 0..4 {
    println!{"Tally of [IE, NS, TF, JP]: {}", tree[i]};
    println!{"{} samples for {}", ensemble[i].1.iter().filter(|&n| *n == 0u8).count(), tree[i].chars().nth(0).unwrap()};
    println!{"{} samples for {}", ensemble[i].1.iter().filter(|&n| *n == 1u8).count(), tree[i].chars().nth(1).unwrap()};
  }

  let model_rf_all_path = Path::new("mbti_rf_all.model");
  let model_svm_all_path = Path::new("mbti_svm_all.model");
  if !model_rf_all_path.exists() || !model_svm_all_path.exists() {
    println!("Generating generic model");
    train(&corpus, &classifiers, "ALL");
  } else {
    println!("Generic models already exists");
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

  let x: DenseMatrix<f64> = DenseMatrix::from_2d_vec(&corpus);
  // These are our target class labels
  let y: Vec<f64> = classifiers.into_iter().map(|x| x as f64).collect();
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
