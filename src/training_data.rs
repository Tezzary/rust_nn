use super::Matrix;
use std::fs;

pub struct TrainingData {
    pub input: Matrix,
    pub output: Matrix
}
pub fn generate_training_data() -> Vec<TrainingData> {

    let mut input_1 = Matrix::zeros(3, 1);
    input_1.set(0, 0, 1.0);
    let mut output_1 = Matrix::zeros(2, 1);
    output_1.set(0, 0, 1.0);
    let data_1 = TrainingData {input: input_1, output: output_1};

    let mut input_2 = Matrix::zeros(3, 1);
    input_2.set(1, 0, 1.0);
    let mut output_2 = Matrix::zeros(2, 1);
    output_2.set(0, 0, 1.0);
    let data_2 = TrainingData {input: input_2, output: output_2};

    let mut input_3 = Matrix::zeros(3, 1);
    input_3.set(2, 0, 1.0);
    let mut output_3 = Matrix::zeros(2, 1);
    output_3.set(1, 0, 1.0);
    let data_3 = TrainingData {input: input_3, output: output_3};

    vec![data_1, data_2, data_3]
}

pub fn read_dataset(filename: &str) -> Vec<TrainingData> {
    let mut data = Vec::new();

    let csv = fs::read_to_string(filename).expect("Failed to read csv file");

    for line in csv.lines() {
        //parsing csv values into readable format
        let values: Vec<&str> = line.split(",").collect();
        let label = values[0].parse::<usize>().unwrap();

        let mut lightness_values = Vec::new();
        for i in 1..values.len() {
            let lightness = values[i].parse::<f32>().unwrap() / 256.0;
            lightness_values.push(lightness);
        }

        //create label matrix
        let mut output_matrix = Matrix::zeros(10, 1);
        output_matrix.set(label, 0, 1.0);

        //create input matrix
        let mut input_matrix = Matrix::zeros(lightness_values.len(), 1);
        for (i, lightness) in lightness_values.iter().enumerate() {
            input_matrix.set(i, 0, *lightness);
        }

        let parsed_data = TrainingData {
            input: input_matrix,
            output: output_matrix,
        };

        data.push(parsed_data);
    }

    data
}