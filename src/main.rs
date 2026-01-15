mod network;
mod matrix;

use matrix::Matrix;
use network::NeuralNetwork;

struct TrainingData {
    input: Matrix,
    output: Matrix
}
fn generate_training_data() -> Vec<TrainingData> {

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

const GENERATIONS: usize = 10000;
const ALPHA: f32 = 0.02;

fn main() {
    let training_data = generate_training_data();

    let dimensions = [3, 2];
    let mut network = NeuralNetwork::new(&dimensions);

    for generation in 0..GENERATIONS {
        let mut sum_loss = 0.0;
        for data in &training_data {
            let data_input = &data.input;
            let data_output = &data.output;

            let output = network.forward_propagate(data_input);

            let loss = network.calculate_loss(data_output);
            //println!("{:?}", output);
            sum_loss += loss;
            
            network.backward_propagate(data_output, ALPHA);
        }

        
        //println!("Generation {} Complete!", generation + 1);
        //println!("Loss: {}", sum_loss / training_data.len() as f32);
    }


}
