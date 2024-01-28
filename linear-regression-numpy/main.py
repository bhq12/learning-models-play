import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


#LEARNING_RATE = 0.00000003
LEARNING_RATE = 0.000003

epsilon = 10**-1


#TODO: Fix mean squared error?

def gradient_descent_step_single_parameter(parameters, training_data_set, labels, parameter_index):
    parameter = parameters[parameter_index]
    gradient_descent_derivative = 0
    for i in range(len(training_data_set)):
        data_point = training_data_set[i]
        data_point_feature = data_point[parameter_index]
        
        label = labels[i]

        hypothesis = perform_linear_hypothesis(parameters, data_point)
        loss_function = (hypothesis - label)

        gradient_descent_derivative += loss_function * data_point_feature

    new_parameter = parameter - LEARNING_RATE * gradient_descent_derivative

    parameters[parameter_index] = new_parameter
    print(f'gradient_descent_derivative: {gradient_descent_derivative}')
    print(f'data_point_feature: {data_point_feature}')
    print(f'parameters: {parameters}')
    return parameters


def gradient_descent_step(parameters, training_data_set, labels):
    for i in range(len(parameters)):
        parameters = gradient_descent_step_single_parameter(parameters, training_data_set, labels, i)

    return parameters

def perform_linear_hypothesis(parameters, data_point):
    result = 0

    for i in range(len(parameters)):
        result += parameters[i] * data_point[i]
    return result

def get_mean_squared_error(training_data_set, labels, parameters):
    total_squared_error = 0
    for i in range(len(training_data_set)):
        data_point = training_data_set[i]
        label = training_data_set[i]

        error = perform_linear_hypothesis(parameters, data_point)
        squared_error = error ** 2
        total_squared_error += squared_error
    return total_squared_error / len(training_data_set)

def main():
    #n: number of features
    #m: number of training data points

    n = 3
    m = 7
    print('Hi')    

    labels = range(1,101) # array size m



    data_set = np.linspace((1,), (100,), num=100)
    theta_0 = np.linspace((1,),(1),num=100)

    data_set = np.concatenate([theta_0, data_set], axis=1)

    random_features = np.random.rand(100,2)

    data_set = np.concatenate([data_set, random_features], axis=1) # 2d array size m x (n + 1)

    print(len(labels))
    print('splitting data')
    training_data_set, test_data_set, training_labels, test_labels = train_test_split(data_set, labels, test_size=0.2, train_size=0.8, random_state=1)
    print('split data')
    print(f'training set: {training_data_set}')

    print(f'test set: {test_data_set}')

    print(f'training labels: {training_labels}')

    print(f'test labels: {test_labels}')


    regression = LinearRegression().fit(training_data_set, training_labels)
    print('score')
    print(regression.score(training_data_set, training_labels))

    print('coefficients')

    print(regression.coef_)

    print('prediction 123')

    print(regression.predict(np.array([[1, 123, 0.2, 0.7]])))
    
if __name__ == '__main__':
    main()
