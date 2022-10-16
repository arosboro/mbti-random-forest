use serde::{Deserialize, Serialize};

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
pub struct MBTI {
    pub indicator: u8,
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
