use crate::myers_briggs::indicator;

pub fn tally(classifiers: &Vec<u8>) {
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
