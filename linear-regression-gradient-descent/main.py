#LEARNING_RATE = 0.00000003
LEARNING_RATE = 0.000003

epsilon = 10**-1


def cost_function():
    print(1)

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
    

    labels = range(1,101) # array size m
    training_data_set = [
        [1, 1, 2.34, 9.99],
        [1, 2, 5.32, 3.8],
        [1, 3, 3.42, 0],
        [1, 4, 6.35, 4.23],
        [1, 5, 9.352, 8.4834],
        [1, 6, 8.385, 8.294],
        [1, 7, 8.385, 2.432],
        [1, 8, 3.853, 8.305],
        [1, 9, 4.859, 8.352],
        [1, 10, 4.852, 6.840],
        [1, 11, 2.34, 9.99],
        [1, 12, 5.32, 3.8],
        [1, 13, 3.42, 0],
        [1, 14, 6.35, 4.23],
        [1, 15, 9.352, 8.4834],
        [1, 16, 8.385, 8.294],
        [1, 17, 8.385, 2.432],
        [1, 18, 3.853, 8.305],
        [1, 19, 4.859, 8.352],
        [1, 20, 4.852, 6.840],
        [1, 21, 2.34, 9.99],
        [1, 22, 5.32, 3.8],
        [1, 23, 3.42, 0],
        [1, 24, 6.35, 4.23],
        [1, 25, 9.352, 8.4834],
        [1, 26, 8.385, 8.294],
        [1, 27, 8.385, 2.432],
        [1, 28, 3.853, 8.305],
        [1, 29, 4.859, 8.352],
        [1, 30, 4.852, 6.840],
        [1, 31, 2.34, 9.99],
        [1, 32, 5.32, 3.8],
        [1, 33, 3.42, 0],
        [1, 34, 6.35, 4.23],
        [1, 35, 9.352, 8.4834],
        [1, 36, 8.385, 8.294],
        [1, 37, 8.385, 2.432],
        [1, 38, 3.853, 8.305],
        [1, 39, 4.859, 8.352],
        [1, 40, 4.852, 6.840],
        [1, 41, 2.34, 9.99],
        [1, 42, 5.32, 3.8],
        [1, 43, 3.42, 0],
        [1, 44, 6.35, 4.23],
        [1, 45, 9.352, 8.4834],
        [1, 46, 8.385, 8.294],
        [1, 47, 8.385, 2.432],
        [1, 48, 3.853, 8.305],
        [1, 49, 4.859, 8.352],
        [1, 50, 4.852, 6.840],
        [1, 51, 2.34, 9.99],
        [1, 52, 5.32, 3.8],
        [1, 53, 3.42, 0],
        [1, 54, 6.35, 4.23],
        [1, 55, 9.352, 8.4834],
        [1, 56, 8.385, 8.294],
        [1, 57, 8.385, 2.432],
        [1, 58, 3.853, 8.305],
        [1, 59, 4.859, 8.352],
        [1, 60, 4.852, 6.840],
        [1, 61, 2.34, 9.99],
        [1, 62, 5.32, 3.8],
        [1, 63, 3.42, 0],
        [1, 64, 6.35, 4.23],
        [1, 65, 9.352, 8.4834],
        [1, 66, 8.385, 8.294],
        [1, 67, 8.385, 2.432],
        [1, 68, 3.853, 8.305],
        [1, 69, 4.859, 8.352],
        [1, 70, 4.852, 6.840],
        [1, 71, 2.34, 9.99],
        [1, 72, 5.32, 3.8],
        [1, 73, 3.42, 0],
        [1, 74, 6.35, 4.23],
        [1, 75, 9.352, 8.4834],
        [1, 76, 8.385, 8.294],
        [1, 77, 8.385, 2.432],
        [1, 78, 3.853, 8.305],
        [1, 79, 4.859, 8.352],
        [1, 80, 4.852, 6.840],
        [1, 81, 2.34, 9.99],
        [1, 82, 5.32, 3.8],
        [1, 83, 3.42, 0],
        [1, 84, 6.35, 4.23],
        [1, 85, 9.352, 8.4834],
        [1, 86, 8.385, 8.294],
        [1, 87, 8.385, 2.432],
        [1, 88, 3.853, 8.305],
        [1, 89, 4.859, 8.352],
        [1, 90, 4.852, 6.840],
        [1, 91, 2.34, 9.99],
        [1, 92, 5.32, 3.8],
        [1, 93, 3.42, 0],
        [1, 94, 6.35, 4.23],
        [1, 95, 9.352, 8.4834],
        [1, 96, 8.385, 8.294],
        [1, 97, 8.385, 2.432],
        [1, 98, 3.853, 8.305],
        [1, 99, 4.859, 8.352],
        [1, 100, .4852, 6.840],
    ] # 2d array size m x (n + 1)

    print(len(labels))
    print(len(training_data_set))
    parameters = [1, 1, 1, 1] # array size (n + 1)



    while True:

        parameters = gradient_descent_step(parameters, training_data_set, labels)
        #print(f'Parameters: {parameters}')
        #mean_squared_error = 5
        mean_squared_error = get_mean_squared_error(training_data_set, labels, parameters)
        print(f'MEAN SQUARED ERROR: {mean_squared_error}')
        if mean_squared_error < epsilon:
            break
        

    
if __name__ == '__main__':
    main()
