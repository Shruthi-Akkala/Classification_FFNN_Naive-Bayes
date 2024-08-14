import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl


class NaiveBayes:
    def __init__(self):
        self.priors = {} 
        self.gaussian_params = {}  
        self.bernoulli_params = {}  
        self.laplace_params = {}  
        self.exponential_params = {}  
        self.multinomial_params = {}  

    def fit(self, X, y):

        """Start of your code."""
        """
        X : np.array of shape (n,10)
        y : np.array of shape (n,)
        Create a variable to store number of unique classes in the dataset.
        Assume Prior for each class to be ratio of number of data points in that class to total number of data points.
        Fit a distribution for each feature for each class.
        Store the parameters of the distribution in suitable data structure, for example you could create a class for each distribution and store the parameters in the class object.
        You can create a separate function for fitting each distribution in its and call it here.
        """
        unique_classes = np.unique(y)
        num_classes = len(unique_classes)

        # prior probabilities for each class
        total_samples = len(y)
        for label in unique_classes:
            class_count = np.sum(y == label)
            self.priors[label] = class_count / total_samples


        
        for label in unique_classes:
            label_data = X[y == label]

            # Gaussian distribution parameters for x1 and x2 (mean and variance)
            self.gaussian_params[label] = (np.mean(label_data[:,[0,1]], axis=0), np.var(label_data[:,[0,1]], axis=0))

            #  Bernoulli distribution parameters for x3 x4 (probability of success)
            self.bernoulli_params[label] = np.mean(label_data[:,[2,3]], axis=0)

            #  Laplace distribution parameters for x5 x6 (mean and scale parameter)
            self.laplace_params[label] = (np.mean(label_data[:,[4,5]], axis=0), np.std(label_data[:,[4,5]], axis=0))

            #  Exponential distribution parameters for x7 x8 (rate parameter)
            self.exponential_params[label] = 1 / np.mean(label_data[:,[6,7]], axis=0)

            #  Multinomial distribution parameters for x9 x10 (probability vector)
            self.multinomial_params[label] = (label_data[:,[8,9]].sum(axis=0) + 1) / (label_data[:,[8,9]].sum() + num_classes)


        """End of your code."""

    def predict(self, X):
        """Start of your code."""
        """
        X : np.array of shape (n,10)

        Calculate the posterior probability using the parameters of the distribution calculated in fit function.
        Take care of underflow errors suitably (Hint: Take log of probabilities)
        Return an np.array() of predictions where predictions[i] is the predicted class for ith data point in X.
        It is implied that prediction[i] is the class that maximizes posterior probability for ith data point in X.
        You can create a separate function for calculating posterior probability and call it here.
        """
        #posterior probabilities for each class

        posterior_probs = np.zeros((X.shape[0], len(self.priors)))
        x_gauss = X[:, [0,1]]
        x_ber = X[:,[2,3]]
        x_lap = X[:,[4,5]]
        x_expo = X[:,[6,7]]
        x_multi = X[:,[8,9]]

        for label, prior in self.priors.items():
            label = int(label)  

            gaussian_likelihood = (
                -0.5 * (np.sum(np.log(2 * np.pi * self.gaussian_params[label][1]))
                + np.sum(((x_gauss - self.gaussian_params[label][0]) ** 2) / self.gaussian_params[label][1], axis=1))
            )
            

            epsilon = 1e-10 
            bernoulli_likelihood = np.sum(
                (x_ber * np.log(self.bernoulli_params[label] + epsilon) + (1 - x_ber) * np.log(1 - self.bernoulli_params[label] + epsilon)),
                axis=1
            )

            
            lap_likelihood = (
                -np.sum(np.log(2 * self.laplace_params[label][1]))
                - np.sum(np.abs(x_lap - self.laplace_params[label][0]) / self.laplace_params[label][1], axis=1)
            )

            
            expo_likelihood = np.sum(
                -self.exponential_params[label] * x_expo, axis=1
            )

            
            epsilon = 1e-10
            multinomial_likelihood = np.prod((x_multi + epsilon) ** (self.multinomial_params[label] + epsilon), axis=1)

            #total log likelihood
            total_log_likelihood = (
                np.log(prior)
                + gaussian_likelihood
                + bernoulli_likelihood
                + lap_likelihood
                + expo_likelihood
                + np.log(multinomial_likelihood)
            )
            posterior_probs[:, label] = total_log_likelihood

        
        predictions = np.argmax(posterior_probs, axis=1)
        return predictions

        """End of your code."""

    def getParams(self):
        """
        Return your calculated priors and parameters for all the classes in the form of dictionary that will be used for evaluation
        Please don't change the dictionary names
        Here is what the output would look like:
        priors = {"0":0.2,"1":0.3,"2":0.5}
        gaussian = {"0":[mean_x1,mean_x2,var_x1,var_x2],"1":[mean_x1,mean_x2,var_x1,var_x2],"2":[mean_x1,mean_x2,var_x1,var_x2]}
        bernoulli = {"0":[p_x3,p_x4],"1":[p_x3,p_x4],"2":[p_x3,p_x4]}
        laplace = {"0":[mu_x5,mu_x6,b_x5,b_x6],"1":[mu_x5,mu_x6,b_x5,b_x6],"2":[mu_x5,mu_x6,b_x5,b_x6]}
        exponential = {"0":[lambda_x7,lambda_x8],"1":[lambda_x7,lambda_x8],"2":[lambda_x7,lambda_x8]}
        multinomial = {"0":[[p0_x9,...,p4_x9],[p0_x10,...,p7_x10]],"1":[[p0_x9,...,p4_x9],[p0_x10,...,p7_x10]],"2":[[p0_x9,...,p4_x9],[p0_x10,...,p7_x10]]}
        """
        priors = {}
        guassian = {}
        bernoulli = {}
        laplace = {}
        exponential = {}
        multinomial = {}

        """Start your code"""
        priors = self.priors
        guassian = self.gaussian_params
        bernoulli = self.bernoulli_params
        laplace = self.laplace_params
        exponential = self.exponential_params
        multinomial = self.multinomial_params



        
        """End your code"""
        return (priors, guassian, bernoulli, laplace, exponential, multinomial)        


def save_model(model,filename="model.pkl"):
    """

    You are not required to modify this part of the code.

    """
    file = open("model.pkl","wb")
    pkl.dump(model,file)
    file.close()

def load_model(filename="model.pkl"):
    """

    You are not required to modify this part of the code.

    """
    file = open(filename,"rb")
    model = pkl.load(file)
    file.close()
    return model

def visualise(data_points,labels):
    """
    datapoints: np.array of shape (n,2)
    labels: np.array of shape (n,)
    """

    plt.figure(figsize=(8, 6))
    plt.scatter(data_points[:, 0], data_points[:, 1], c=labels, cmap='viridis')
    plt.colorbar()
    plt.title('Generated 2D Data from 5 Gaussian Distributions')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()


def net_f1score(predictions, true_labels):
    """Calculate the multclass f1 score of the predictions.
    For this, we calculate the f1-score for each class 

    Args:
        predictions (np.array): The predicted labels.
        true_labels (np.array): The true labels.

    Returns:
        float(list): The f1 score of the predictions for each class
    """

    def precision(predictions, true_labels, label):
        """Calculate the multclass precision of the predictions.
        For this, we take the class with given label as the positive class and the rest as the negative class.

        Args:
            predictions (np.array): The predicted labels.
            true_labels (np.array): The true labels.

        Returns:
            float: The precision of the predictions.
        """
        """Start of your code."""
        true_positives = np.sum((predictions == label) & (true_labels == label))
        false_positives = np.sum((predictions == label) & (true_labels != label))
        precision = true_positives / (true_positives + false_positives + 1e-10) # Add a small epsilon to avoid division by zero
        return precision


        
        """End of your code."""
        


    def recall(predictions, true_labels, label):
        """Calculate the multclass recall of the predictions.
        For this, we take the class with given label as the positive class and the rest as the negative class.
        Args:
            predictions (np.array): The predicted labels.
            true_labels (np.array): The true labels.

        Returns:
            float: The recall of the predictions.
        """
        """Start of your code."""
        true_positives = np.sum((predictions == label) & (true_labels == label))
        false_negatives = np.sum((predictions != label) & (true_labels == label))
        recall = true_positives / (true_positives + false_negatives + 1e-10)  # Add a small epsilon to avoid division by zero
        return recall




        """End of your code."""
        

    def f1score(predictions, true_labels, label):
        """Calculate the f1 score using it's relation with precision and recall.

        Args:
            predictions (np.array): The predicted labels.
            true_labels (np.array): The true labels.

        Returns:
            float: The f1 score of the predictions.
        """

        """Start of your code."""
        precision_value = precision(predictions, true_labels, label)
        recall_value = recall(predictions, true_labels, label)
        f1 = 2 * (precision_value * recall_value) / (precision_value + recall_value + 1e-10)



        """End of your code."""
        return f1
    

    f1s = []
    for label in np.unique(true_labels):
        f1s.append(f1score(predictions, true_labels, label))
    return f1s

def accuracy(predictions,true_labels):
    """

    You are not required to modify this part of the code.

    """
    return np.sum(predictions==true_labels)/predictions.size



if __name__ == "__main__":
    """

    You are not required to modify this part of the code.

    """

    # Load the data
    train_dataset = pd.read_csv('./data/train_dataset.csv',index_col=0).to_numpy()
    validation_dataset = pd.read_csv('./data/validation_dataset.csv',index_col=0).to_numpy()

    # Extract the data
    train_datapoints = train_dataset[:,:-1]
    train_labels = train_dataset[:, -1]
    validation_datapoints = validation_dataset[:, 0:-1]
    validation_labels = validation_dataset[:, -1]

    # Visualize the data
    # visualise(train_datapoints, train_labels, "train_data.png")

    # Train the model
    model = NaiveBayes()
    model.fit(train_datapoints, train_labels)

    # Make predictions
    train_predictions = model.predict(train_datapoints)
    validation_predictions = model.predict(validation_datapoints)

    # Calculate the accuracy
    train_accuracy = accuracy(train_predictions, train_labels)
    validation_accuracy = accuracy(validation_predictions, validation_labels)

    # Calculate the f1 score
    train_f1score = net_f1score(train_predictions, train_labels)
    validation_f1score = net_f1score(validation_predictions, validation_labels)

    # Print the results
    print('Training Accuracy: ', train_accuracy)
    print('Validation Accuracy: ', validation_accuracy)
    print('Training F1 Score: ', train_f1score)
    print('Validation F1 Score: ', validation_f1score)

    # Save the model
    save_model(model)

    # Visualize the predictions
    # visualise(validation_datapoints, validation_predictions, "validation_predictions.png")

