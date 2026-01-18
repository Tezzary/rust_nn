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
        for i in 1..dimensions.len() {
            network.biases.push(Matrix::populate_random(dimensions[i], 1, -0.01, 0.01));
        }

        //initiate weights
        for i in 0..dimensions.len() - 1{
            let next_dim_size = dimensions[i + 1];
            let current_dim_size = dimensions[i];

            let weights = Matrix::populate_random(next_dim_size, current_dim_size, 0.0, 0.01);

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
    //standard MSE Loss SUM((x1 - x2)^2)
    pub fn calculate_loss(&self, data_outputs: &Matrix) -> f32 {
        let output_layer = &self.neurons[self.neurons.len() - 1];

        let mut loss = 0.0;

        for y in 0..output_layer.get_rows() {
            loss += (output_layer.get(y, 0) - data_outputs.get(y, 0)).powi(2);
        }

        loss
    }

    pub fn loss_derivative(&self, data_outputs: &Matrix) -> Matrix {
        let output_layer = &self.neurons[self.neurons.len() - 1];

        let mut derivatives = Matrix::zeros(output_layer.get_rows(), 1);

        //TODO optimise to be a matrix subtraction followed by a scalar multiply of 2
        for y in 0..output_layer.get_rows() {
            let derivative = 2.0 * (output_layer.get(y, 0) - data_outputs.get(y, 0));
            derivatives.set(y, 0, derivative);
        }

        derivatives
    }
   
    pub fn backward_propagate(&mut self, data_outputs: &Matrix, alpha: f32) {
        let loss_derivatives = self.loss_derivative(data_outputs);

        let neuron_count = self.neurons.len();

        let mut chained_derivatives = self.neurons.clone();
        chained_derivatives[neuron_count - 1] = loss_derivatives;

        let final_index = self.neurons.len() - 1;

        for i in 0..neuron_count - 1 {
            let neuron_layer = &self.neurons[final_index - i];
            let relu_derivative = neuron_layer.relu_derivative();
            let chained_derivative = &chained_derivatives[final_index - i];
            let chained_derivative = chained_derivative.pointwise_multiply(&relu_derivative);

            let previous_neuron_layer = &self.neurons[final_index - i - 1];
            let previous_weight_layer = &self.weights[final_index - i - 1];
            let previous_bias_layer = &self.biases[final_index - i - 1];

            //find derivatives of weights given by d/n1 x (n0)^T
            let derivative_weights = chained_derivative.multiply(&previous_neuron_layer.transpose());

            //find derivatives of biases given by d/n1 x 1
            //println!("{:?}", chained_derivatives);
            //println!("{:?}", previous_bias_layer);
            //println!("{:?}", self.biases);
            let derivative_biases = &chained_derivative;

            //find derivatives of neurons in previous layer for chaining given by ((d/n1)^T x w)^T
            let transposed_chained_derivative = chained_derivative.transpose();
            let derivative_neurons = transposed_chained_derivative.multiply(previous_weight_layer).transpose();

            self.weights[final_index - i - 1] = previous_weight_layer.subtract(&derivative_weights.scalar_multiply(alpha));
            self.biases[final_index - i - 1] = previous_bias_layer.subtract(&derivative_biases.scalar_multiply(alpha));
            chained_derivatives[final_index - i - 1] = derivative_neurons;
        }
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