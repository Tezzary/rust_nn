mod network;
mod matrix;
mod training_data;

use matrix::Matrix;
use network::NeuralNetwork;


const GENERATIONS: usize = 20;
const ALPHA: f32 = 0.0005;

fn is_correct(real_output: &Matrix, network_output: &Matrix) -> bool {
    let mut max = -1.0;
    let mut max_y = 0;
    let mut actual_y = 0;
    for y in 0..network_output.get_rows() {
        let value = network_output.get(y, 0);
        if value > max {
            max = value;
            max_y = y;
        }
        if real_output.get(y, 0) == 1.0 {
            actual_y = y;
        }
    }
    if actual_y == max_y {
        return true;
    }
    return false;

}

fn main() {
    let training_data = training_data::read_dataset("data/mnist_train.csv");

    println!("loaded data");
    let dimensions = [784, 256, 128, 10];
    let mut network = NeuralNetwork::new(&dimensions);

    for generation in 0..GENERATIONS {
        let mut sum_loss = 0.0;
        let mut successes = 0;
        for data in &training_data {
            let data_input = &data.input;
            let data_output = &data.output;

            let output = network.forward_propagate(data_input);

            if is_correct(&data_output, &output) {
                successes += 1;
            }
            

            let loss = network.calculate_loss(data_output);
            //println!("{:?}", output);
            sum_loss += loss;
            
            network.backward_propagate(data_output, ALPHA);
        }

        
        println!("Generation {} Complete!", generation + 1);
        println!("Loss: {}", sum_loss / training_data.len() as f32);
        println!("Correctly Guessed: {:.2}%", (successes * 100) as f32 / training_data.len() as f32);
    }

    println!("Completed Training, Beginning Testing...");

    let test_data = training_data::read_dataset("data/mnist_test.csv");

    let mut successes = 0;
    for data in &test_data{
        let data_input = &data.input;
        let data_output = &data.output;

        let output = network.forward_propagate(data_input);

        if is_correct(&data_output, &output) {
            successes += 1;
        }
    }

    println!("Correctly Guessed: {:.2}% of Test Data Correct!", (successes * 100) as f32 / test_data.len() as f32);
}
