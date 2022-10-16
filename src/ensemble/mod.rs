pub fn build_sets(
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
