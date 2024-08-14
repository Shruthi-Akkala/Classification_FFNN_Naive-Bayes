import numpy as np

class LogisticRegression:
    def __init__(self):
        """
        Initialize `self.weights` properly. 
        Recall that for binary classification we only need 1 set of weights (hence `num_classes=1`).
        We have given the default zero intialization with bias term (hence the `d+1`).
        You are free to experiment with various other initializations including random initialization.
        Make sure to mention your initialization strategy in your report for this task.
        """
        self.num_classes = 1 # single set of weights needed
        self.d = 2 # input space is 2D. easier to visualize
        self.v = 0
        self.weights = np.ones((self.d+1, self.num_classes))
    
    def preprocess(self, input_x):
        """
        Preprocess the input any way you seem fit.
        """
        return input_x

    def sigmoid(self, x):
        """
        Implement a sigmoid function if you need it. Ignore otherwise.
        """
        return 1/(1+np.exp(-x))

    def calculate_loss(self, input_x, input_y):
        """
        Arguments:
        input_x -- NumPy array with shape (N, self.d) where N = total number of samples
        input_y -- NumPy array with shape (N,)
        Returns: a single scalar value corresponding to the loss.
        """
        samples = input_x.shape[0]
        input_x = np.hstack((np.ones((samples,1)),input_x))
        z = np.dot(input_x, self.weights)
        y_pred = self.sigmoid(z)
        loss= -(np.mean(np.dot(input_y, np.log(y_pred)) + np.dot(1-input_y,np.log(1-y_pred))))/samples
        return loss

    def calculate_gradient(self, input_x, input_y):
        """
        Arguments:
        input_x -- NumPy array with shape (N, self.d) where N = total number of samples
        input_y -- NumPy array with shape (N,)
        Returns: the gradient of loss function wrt weights.
        Ensure that gradient.shape == self.weights.shape.
        """
        N = input_x.shape[0]
        input_x = np.hstack((np.ones((N, 1)), input_x))  
        z = np.dot(input_x, self.weights)
        predictions = self.sigmoid(z)
        y_trans = input_y.reshape([150,1])
        
        gradient = np.dot(input_x.T, (predictions - y_trans))*(1/ N)
        # print(gradient)
        return gradient

     
    def update_weights(self, grad, learning_rate, momentum):
        self.v = self.v*momentum - learning_rate*grad
        self.weights = self.weights + self.v 
    
    def get_prediction(self, input_x):
        m,n = input_x.shape
        input_x = np.hstack((np.ones((m, 1)), input_x))
        predictions = np.zeros(m)
        for i in range(m):
            z = np.dot(input_x[i], self.weights)
            for j in range(n):
                z += 0
            f_wb = self.sigmoid(z)
            predictions[i] = 1 if f_wb>0.5 else 0
        return predictions


class LinearClassifier:
    def __init__(self):
        self.num_classes = 3
        self.d = 4
        self.v = 0
        self.weights = np.zeros((self.d + 1, self.num_classes))
    
    def preprocess(self, train_x):
        #train_x[:,2] = 0
        return train_x
    
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def calculate_loss(self, input_x, input_y):
        N = input_x.shape[0]
        input_x = np.hstack((np.ones((N, 1)), input_x))
        scores = np.dot(input_x, self.weights)
        probs = self.sigmoid(scores)
        loss = -np.mean(np.log(probs[np.arange(N), input_y]))
        return loss
    
    def calculate_gradient(self, input_x, input_y):
        N = input_x.shape[0]
        input_x = np.hstack((np.ones((N, 1)), input_x))
        scores = np.dot(input_x, self.weights)
        probs = self.sigmoid(scores)
        probs[np.arange(N), input_y] -= 1
        gradient = np.dot(input_x.T, probs) / N
        return gradient
    
    def update_weights(self, grad, learning_rate, momentum):
        self.v = self.v*momentum - learning_rate*grad
        self.weights = self.weights + self.v
    
    def get_prediction(self, input_x):
        input_x = np.hstack((np.ones((input_x.shape[0], 1)), input_x))
        scores = np.dot(input_x, self.weights)
        predictions = np.argmax(scores, axis=1)
        return predictions