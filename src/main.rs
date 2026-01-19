mod network;
mod matrix;
mod training_data;

use matrix::Matrix;
use network::NeuralNetwork;

use std::fs;

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

    if fs::exists("results").expect("failed to check results exists") {
        fs::remove_dir_all("results").expect("Failed to clear results directory");
    }
    fs::create_dir_all("results/successes").unwrap();
    fs::create_dir_all("results/fails").unwrap();

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
    let mut fails = 0;


    for data in &test_data{
        let data_input = &data.input;
        let data_output = &data.output;

        let output = network.forward_propagate(data_input);

        let mut data_string = data.to_string();
        let resized_output = output.resize(1, 10);
        data_string.push_str(&("Network Output: ".to_string() + &resized_output.to_string()));

        let mut largest_output = -1.0;
        let mut largest_index = 0;
        for x in 0..resized_output.get_cols() {
            let value = resized_output.get(0, x);
            if value > largest_output {
                largest_output = value;
                largest_index = x;
            }
        }
        data_string.push_str(&format!("Network Label: {}\n", largest_index));

        if is_correct(&data_output, &output) {
            fs::write(format!("results/successes/{}.txt", successes), data_string).expect("Failed to write to successes");
            successes += 1;
        }
        else {
            fs::write(format!("results/fails/{}.txt", fails), data_string).expect("Failed to write to fails");
            fails += 1;
        }
    }

    println!("Correctly Guessed: {:.2}% of Test Data Correct!", (successes * 100) as f32 / test_data.len() as f32);
}
