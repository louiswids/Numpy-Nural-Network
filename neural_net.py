# followed Bert Gollnick's Udemy Tutorial
import numpy as np
class neural_net_scratch:
    def __init__(self, lr, X_train, y_train, X_test, y_test):
        self.lr = lr
        self.W = np.random.randn(X_train.shape[1]) # weights with dimension of X_train columns
        self.b = np.random.randn()
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.L_train = []
        self.L_test = []

    def activation (self, x):
        return (1 / (1 + np.exp(-x)))

    def d_activation (self, x):
        res = self.activation(x)
        return (res * (1 - res))

    def forward_pass (self, X):
        hidden_1 = np.dot (X, self.W) + self.b
        activation_1 = self.activation(hidden_1)
        return activation_1
    def backward_pass (self, X, y_true):
        hidden_1 = np.dot(X, self.W) + self.b
        y_pred = self.forward_pass(X)
        dL_dpred = 2 * (y_pred - y_true)
        dpred_dhidden1 = self.d_activation(hidden_1)
        dhidden1_db = 1
        dhidden1_dW = X 

        #use chain rule
        dL_db = dL_dpred * dpred_dhidden1 * dhidden1_db
        dL_dW = dL_dpred * dpred_dhidden1 * dhidden1_dW
        
        return dL_dW, dL_db
        
    def optimizer (self, dL_dW, dL_db ):
        self.W = self.W - (self.lr * dL_dW)
        self.b = self.b - (self.lr * dL_db)
    def train (self, ITERATIONS):
        for i in range (ITERATIONS):
            rand_position = np.random.randint(len(self.X_train))
            y_train_true = self.y_train[rand_position]
            y_train_pred = self.forward_pass(self.X_train[rand_position])

            L_train = np.sum(np.square(y_train_true - y_train_pred))
            self.L_train.append(L_train)

            # gradients
            dL_dW, dL_db = self.backward_pass(self.X_train[rand_position], 
                                              self.y_train[rand_position])
            self.optimizer(dL_dW, dL_db)

            # Error calc
            L_sum = 0
            for j in range (len (self.X_test)):
                y_true = self.y_test[j]
                y_pred = self.forward_pass(self.X_test[j])
                L_sum = np.sum(np.square(y_true - y_pred))
                
            self.L_test.append(L_sum)
            
        return "Done Training"

    