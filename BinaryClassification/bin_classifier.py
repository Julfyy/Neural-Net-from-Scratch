import numpy as np


class Classifier:
    def __init__(self, input_dim, output_dim, hidden_dim):
        self.input = np.zeros(input_dim)
        self.hidden = np.ones(hidden_dim)
        self.output = np.zeros(output_dim)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.weights1 = np.random.randn(hidden_dim, input_dim)
        self.weights2 = np.random.randn(output_dim, hidden_dim)
        
        #z is a sum of weigth products for every neuron, z[i] = w[i,j] * a[j]
        #For neurons i use next formula:  a[i] = sigmoid(z[i])
        self.z1 = np.zeros(hidden_dim)
        self.z2 = np.zeros(output_dim)
       
        #y_hat is a desired output
        #In classification problem it looks like that [0,1,0,0,0,0,0] for number 1, etc
        self.y_hat = np.zeros(output_dim)
        
    def sigmoid(self, x):
        return (1 / (1 + np.exp(-x)))
    
    
    #For cost function I use sum of squared errors
    def cost(self, y, y_hat):
        cost = 0;
        for i in range(len(y)):
            cost += (y_hat[i] - y[i])**2
        return cost
    
    
    def sigmoid_deriv(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))    
        
    
    
    '''
    Feeding forward is a process of calculating values for every neuron.
    Formula: a[i] = sigmoid(z[i]), where z[i] = w[i][j] * a[j],
             and a[j] is a neuron of a previous layer.
    '''
    def feed_forward(self):
        #Calculating hidden neurons
        for j in range(self.hidden_dim):
            self.z1[j] = np.dot(self.weights1[j], self.input)
            self.hidden[j] = self.sigmoid(self.z1[j])
        
        #Calculating output neurons
        for i in range(self.output_dim):
            self.z2[i] = np.dot(self.weights2[i], self.hidden)
            self.output[i] = self.sigmoid(self.z2[i])
        
    
    '''
    Back propagation is the core process for training.
    Here I use gradient descent method, to optimize weigths
    
        For back propagation we need to find a negative gradient of the cost
        function, that shows us the direction of the steepest descent.
        
        If we calculate this gradient in a single dot (weights at the moment),
        we'll find values, showing how much we need to increase or 
        decrease each weight.
        
        Also we need a learning rate koeficient to avoid overstepping.
        
        The cost function has 2 groups of variables C(w1,w2).
        Each w_n is a matrix of some numbers.
        The sum of all weights is 50 (for this neural network).
        
        So the gradient of the cost function is a 50 dimensional vector, each
        component is a partial derivative with respect to some weigth.
        
        Gradient descent means W_j_i -= a * d(Cost)/d(W_j_i)
    '''
    
    def back_propagation(self):
    
        lr = 0.5 #Learning rate
        
        #Adjusting first weigths        
        for j in range(self.hidden_dim):
            i_sum = 0
            for i in range(self.output_dim):
                i_sum += self.weights2[i][j] * self.sigmoid_deriv(self.z2[i]) * 2 * (self.output[i] - self.y_hat[i] )
            for k in range(self.input_dim):
                w1_deriv = self.input[k] * self.sigmoid_deriv(self.z1[j]) * i_sum
                self.weights1[j][k] -= lr * w1_deriv  
                
        #Adjusting second weights
        for i in range(self.output_dim):
            for j in range(self.hidden_dim):
                w2_deriv = 2 * (self.output[i] - self.y_hat[i]) * self.sigmoid_deriv(self.z2[i]) * self.hidden[j]
                self.weights2[i][j] -= lr * w2_deriv
                
              
    def single_prediction(self, x):
        self.input = x
        self.feed_forward()
        print(np.argmax(self.output))
                
                
    def fit_data(self, x_train, y_train, epochs):
        cost = []
        accuracy = []
        
        for epoch in range(1, epochs + 1):
            print('Epoch number', epoch)
            for image in range(len(x_train)):#For every image in batch
                
                print('\n=====№', image)
                
                #Initialize y_hat
                self.y_hat[y_train[image]] = 1
                
                print("Number is: ", y_train[image])

                #initialize an input layer
                self.input = x_train[image]
                
                #Feed every image forward
                self.feed_forward()
                
                #Calculate cost and accuracy
                cost.append(self.cost(self.output, self.y_hat))
                print("Cost: ", cost[-1])
                accuracy.append(round(100 - (cost[-1] * 100 / self.output_dim)))
                print('Accuracy is: ', accuracy[-1], '%')
                                
                self.back_propagation()
                
                print('Output is a number: s', np.argmax(self.output))
                self.y_hat = np.zeros(self.output_dim)
        print('\nAverage cost is: ', np.mean(cost))
        print('Average accuracy is ', np.mean(accuracy), '%')
        
                
                
        
                
        