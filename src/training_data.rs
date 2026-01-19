use super::Matrix;
use std::fs;

pub struct TrainingData {
    pub input: Matrix,
    pub output: Matrix
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

impl TrainingData {
    pub fn to_string(&self) -> String {
        let resized_input = self.input.resize(28, 28);
        let mut data_print_out = resized_input.to_string_image();

        let resized_output = self.output.resize(1, 10);
        data_print_out.push_str(&("Correct Output: ".to_string() + &resized_output.to_string()));
        for x in 0..resized_output.get_cols() {
            let value = resized_output.get(0, x);
            if value != 0.0 {
                data_print_out.push_str(&format!("Correct Label: {}\n", x));
            }
        }
        data_print_out
    }
}