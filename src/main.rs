use csv::Error;
use serde::Deserialize;
// DenseMatrix wrapper around Vec
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
// Random Forest
use smartcore::ensemble::random_forest_regressor::RandomForestRegressor;
// Model performance
use smartcore::metrics::mean_squared_error;
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

#[derive(Copy, Clone)]
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

struct Sample {
  indicator: MBTI,
  posts: Vec<Post>,
}

fn tokenize(post: &str) -> Post {
  let tokenizer = VTextTokenizerParams::default().lang("en").build().unwrap();
  let tokens: Vec<String> = tokenizer.tokenize(post).map(|s| s.to_owned()).collect();
  tokens
}

fn main() -> Result<(), Error> {
  let mut reader = csv::Reader::from_path("/Users/arosboro/projects/mbti-random-forest/mbti_1.csv").unwrap();
  let mut training_set: Vec<Sample> = Vec::new();
  for row in reader.deserialize::<Row>() {
    match row {
      Ok(row) => {
        let mut sample = Sample {
          indicator: MBTI::from_string(&row.label),
          posts: Vec::new(),
        };
        for post in row.posts.split("|||") {
          sample.posts.push(tokenize(post));
        }
        // println!("{}: {} posts", sample.indicator.to_string(), sample.posts.len());
        training_set.push(sample)
      },
      Err(e) => println!("Error: {}", e),
    }
  }

  let mut x_set: Vec<Post> = Vec::new();
  let mut y_set: Vec<MBTI> = Vec::new();
  for sample in training_set {
    for post in sample.posts {
      x_set.push(post);
      y_set.push(sample.indicator);
    }
  }

  let mut dictionary: Vec<String> = Vec::new();
  for post in x_set.clone() {
    for token in post {
      if !dictionary.contains(&token) {
        dictionary.push(token);
      }
    }
  }

  let mut x_matrix: Vec<Vec<f64>> = Vec::new();
  for post in x_set {
    let mut matrix: Vec<f64> = Vec::new();
    for token in post {
      let index: f64 = dictionary.iter().position(|x| x == &token).unwrap() as f64;
      matrix.push(index);
    }
    x_matrix.push(matrix);
  }

  let mut y_matrix: Vec<f64> = Vec::new();
  for indicator in y_set {
    let index: f64 = indicator.indicator as f64;
    y_matrix.push(index);
  }

  let x: DenseMatrix<f64> = DenseMatrix::from_2d_vec(&x_matrix);
  // These are our target class labels
  let y: Vec<f64> = y_matrix;
  // Split dataset into training/test (80%/20%)
  let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true);
  // Random Forest
  let y_hat_rf = RandomForestRegressor::fit(&x_train, &y_train, Default::default())
      .and_then(|rf| rf.predict(&x_test)).unwrap();
  // Calculate test error
  println!("MSE: {}", mean_squared_error(&y_test, &y_hat_rf));

  Ok(())
}
