use crate::matrix::Matrix;

#[derive(Debug)]
pub struct NeuralNetwork {
    weights: Vec<Matrix>,
    neurons: Vec<Matrix>,
    biases: Vec<Matrix>,
}

impl NeuralNetwork {
    pub fn new(dimensions: &[usize]) -> NeuralNetwork {
        let mut network = NeuralNetwork {
            neurons: Vec::new(),
            weights: Vec::new(),
            biases: Vec::new(),
        };

        //initiate neurons
        for dim in dimensions {
            network.neurons.push(Matrix::zeros(*dim, 1));
        }

        //initiate biases
        //TODO: RANDOM RATHER THAN 0's
        for i in 1..dimensions.len() {
            network.biases.push(Matrix::populate_random(dimensions[i], 1));
        }

        //initiate weights
        //TODO: RANDOM RATHER THAN 0's
        for i in 0..dimensions.len() - 1{
            let next_dim_size = dimensions[i + 1];
            let current_dim_size = dimensions[i];

            let weights = Matrix::populate_random(next_dim_size, current_dim_size);

            network.weights.push(weights);
        }

        network
    }

    pub fn forward_propagate(&mut self, inputs: &Matrix) -> Matrix {
        assert_eq!(inputs.get_rows(), self.neurons[0].get_rows());
        assert_eq!(inputs.get_cols(), self.neurons[0].get_cols());

        self.neurons[0] = inputs.clone();

        for i in 0..self.neurons.len() - 1 {
            //y = ax + b
            let ax = self.weights[i].multiply(&self.neurons[i]);
            let mut ax_b = ax.add(&self.biases[i]);
            ax_b.relu();
            self.neurons[i + 1] = ax_b;
        }

        self.neurons[self.neurons.len() - 1].clone()
    }

    //returns the loss of the function
    pub fn backward_propagate(&mut self, data_outputs: &Matrix) -> f32 {
        0.0
    }
}

#[cfg(test)]
mod test {

    use super::*;

    #[test]
    fn create_network() {

        let dimensions = [2, 3, 2, 3, 1];
        let mut network = NeuralNetwork::new(&dimensions);

        let mut inputs = Matrix::zeros(2, 1);

        inputs.set(0, 0, 1.0);
        inputs.set(1, 0, 2.0);

        let output = network.forward_propagate(&inputs);

        //println!("{:?}", network.neurons);
        //println!("{:?}", network.biases);
        //println!("{:?}", network.weights);
        //println!("{:?}", output);
    }
}