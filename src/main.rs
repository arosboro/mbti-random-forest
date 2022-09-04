use std::fs::File;
use std::io::{Write, Read};
use csv::Error;
use serde::{
  Serialize, 
  Deserialize
};
// DenseMatrix wrapper around Vec
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
// Random Forest
use smartcore::ensemble::random_forest_regressor::RandomForestRegressor;
// Model performance
use smartcore::metrics::{mean_squared_error, accuracy};
use smartcore::model_selection::train_test_split;
use vtext::tokenize::*;

#[derive(Debug, Deserialize)]
struct Row {
  label: String,
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

#[derive(Debug, Serialize, Deserialize)]
struct Sample {
  indicator: MBTI,
  posts: Vec<Post>,
}

fn tokenize(post: &str) -> Post {
  let tokenizer = VTextTokenizerParams::default().lang("en").build().unwrap();
  let tokens: Vec<String> = tokenizer.tokenize(post).map(|s| s.to_lowercase().to_owned()).collect();
  tokens
}

fn load_data() -> Vec<Sample> {
  let f = std::fs::OpenOptions::new().write(false).create(false).truncate(false).open("samples.bincode");
  if let Ok(mut f) = f {
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).unwrap();
    let samples: Vec<Sample> = bincode::deserialize(&buf).unwrap();
    return samples;
  }
  else {
    let mut reader = csv::Reader::from_path("./mbti_1.csv").unwrap();
    let mut samples: Vec<Sample> = Vec::new();
    for row in reader.deserialize::<Row>() {
      match row {
        Ok(row) => {
          let mut sample: Sample = Sample {
            indicator: MBTI::from_string(&row.label),
            posts: Vec::new(),
          };
          for post in row.posts.split("|||") {
            sample.posts.push(tokenize(post));
          }
          // println!("{}: {} posts", sample.indicator.to_string(), sample.posts.len());
          samples.push(sample)
        },
        Err(e) => println!("Error: {}", e),
      }
    }
    let f = std::fs::OpenOptions::new().write(true).create(true).truncate(true).open("samples.bincode");
    let samples_bytes = bincode::serialize(&samples).unwrap();
    f.and_then(|mut f| f.write_all(&samples_bytes)).expect("Failed to write samples");
    return samples;
  }
}

fn normalize(training_set: &Vec<Sample>) -> (Vec<Vec<f64>>, Vec<u8>) {
  let mut x_set: Vec<Post> = Vec::new();
  let mut y_set: Vec<MBTI> = Vec::new();

  for sample in training_set {
    for post in &sample.posts {
      x_set.push(post.to_vec());
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

  let f = std::fs::OpenOptions::new().write(false).create(false).truncate(false).open("dictionary.bincode");
  let dictionary: Vec<String> = {
    if let Ok(mut f) = f {
      let mut buf = Vec::new();
      f.read_to_end(&mut buf).unwrap();
      let dictionary: Vec<String> = bincode::deserialize(&buf).unwrap();
      dictionary
    }
    else {
      // Create a dictionary indexing unique tokens.
      let mut dictionary: Vec<String> = Vec::new();
      for post in x_set.clone() {
        for token in post {
          if !dictionary.contains(&token) {
            dictionary.push(token);
          }
        }
      }
      println!("Saving dictionary...");
      let f = std::fs::OpenOptions::new().write(true).create(true).truncate(true).open("dictionary.bincode");
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
      let index: f64 = dictionary.iter().position(|x| x == &token).unwrap() as f64;
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

  println!("Saving matrices...");
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

fn tally(y_matrix: &Vec<u8>) {
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


fn train(x_matrix: &Vec<Vec<f64>>, y_matrix: &Vec<u8>) {
  let x: DenseMatrix<f64> = DenseMatrix::from_2d_vec(&x_matrix);
  // These are our target class labels
  let y: Vec<f64> = y_matrix.into_iter().map(|x| *x as f64).collect();
  // Split bag into training/test (80%/20%) TODO: Does it mater we do this every training cycle?
  let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true);

  // Random Forest
  RandomForestRegressor::fit(&x_train, &y_train, Default::default())
    .into_iter()
    .for_each(|rf| {
      let y_pred = rf.predict(&x_test).unwrap();
      let accuracy = accuracy(&y_test, &y_pred);
      println!("Accuracy: {}", accuracy);
      let rf_bytes = bincode::serialize(&rf).expect("Can not serialize the model");
      let f = std::fs::OpenOptions::new().write(true).create(true).truncate(true).open("mbti_rf.model");
      f.and_then(|mut f| f.write_all(&rf_bytes))
        .expect("Can not persist model");
    });
  
  // Load the Model
  let rf: RandomForestRegressor<f64> = {
    let mut buf: Vec<u8> = Vec::new();
    File::open("mbti_rf.model")
      .and_then(|mut f| f.read_to_end(&mut buf))
      .expect("Can not load model");
    bincode::deserialize(&buf).expect("Can not deserialize the model")
  };
    
  let y_hat_rf = rf.predict(&x_test).unwrap();
  // Calculate test error
  println!("MSE: {}", mean_squared_error(&y_test, &y_hat_rf));
}

fn main() -> Result<(), Error> {
  let training_set: Vec<Sample> = load_data();
  let (x_matrix, y_matrix) = normalize(&training_set);
  tally(&y_matrix);
  train(&x_matrix, &y_matrix);

  Ok(()) 
}
