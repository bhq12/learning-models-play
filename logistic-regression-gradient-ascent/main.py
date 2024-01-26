from math import exp

LEARNING_RATE = 0.0003

def gradient_descent_step_single_parameter(parameters, training_data_set, labels, parameter_index):
    parameter = parameters[parameter_index]
    gradient_descent_derivative = 0
    for i in range(len(training_data_set)):
        data_point = training_data_set[i]
        data_point_feature = data_point[parameter_index]
        
        label = labels[i]

        hypothesis = perform_logistic_hypothesis(parameters, data_point)
        loss_function = (hypothesis - label)

        gradient_descent_derivative += loss_function * data_point_feature

    gradient_descent_derivative = gradient_descent_derivative / len(training_data_set)
    
    #Plus instead of minus as we're doing gradient ascent
    # to find the maxima rather than the minima
    # and then the loss_function above is inverted compared to linear regression
    new_parameter = parameter - LEARNING_RATE * gradient_descent_derivative

    parameters[parameter_index] = new_parameter
    # print(f'gradient_descent_derivative: {gradient_descent_derivative}')
    # print(f'data_point_feature: {data_point_feature}')
    # print(f'parameters: {parameters}')
    return parameters


def gradient_descent_step(parameters, training_data_set, labels):
    for i in range(len(parameters)):
        parameters = gradient_descent_step_single_parameter(parameters, training_data_set, labels, i)

    return parameters

def perform_logistic_hypothesis(parameters, data_point):
    exponent = 0
    # Theta (transposed) times x (data)
    for i in range(len(parameters)):
        # print(f'parameters: ' + str(parameters))
        # print(f'data_point: ' + str(data_point))
        # print(f'i: ' + str(i))
        exponent += parameters[i] * data_point[i]
    
    #Sigmoid function raises e to the power of negative theta.x
    exponent = 0 - exponent

    result = 1 / (1 + exp(exponent))

    return result

def main():
    #n: number of features
    #m: number of training data points

    n = 3
    m = 100

    labels = []

    for i in range(45):
        labels.append(0)

    labels.append(1)
    labels.append(0)
    labels.append(1)
    labels.append(0)
    labels.append(1)
    labels.append(0)
    labels.append(1)
    labels.append(0)
    labels.append(1)
    labels.append(0)

    for i in range(45):
        labels.append(1)

    

    training_data_set = []
    for i in range(1,101):
        training_data_set.append([1, i])

    print(len(labels))
    print(training_data_set)
    parameters = [0.001, 0.001] # array size (n + 1)

    iterations = 0

    while True:
        parameters = gradient_descent_step(parameters, training_data_set, labels)
        iterations += 1
        if iterations > 1000000:
            break
        print(f'iteration: {iterations}')
        print(f'parameters: {parameters}')
        print('-200 estimate: ' + str(perform_logistic_hypothesis(parameters, [1, -200])))
        print('-50 estimate: ' + str(perform_logistic_hypothesis(parameters, [1, -50])))
        print('1 estimate: ' + str(perform_logistic_hypothesis(parameters, [1, 1])))
        print('25 estimate: ' + str(perform_logistic_hypothesis(parameters, [1, 25])))

        print('35 estimate: ' + str(perform_logistic_hypothesis(parameters, [1, 35])))
        print('40 estimate: ' + str(perform_logistic_hypothesis(parameters, [1, 40])))
        print('45 estimate: ' + str(perform_logistic_hypothesis(parameters, [1, 45])))
        print('50 estimate: ' + str(perform_logistic_hypothesis(parameters, [1, 50])))
        print('55 estimate: ' + str(perform_logistic_hypothesis(parameters, [1, 55])))
        print('60 estimate: ' + str(perform_logistic_hypothesis(parameters, [1, 60])))
        print('65 estimate: ' + str(perform_logistic_hypothesis(parameters, [1, 65])))

        print('75 estimate: ' + str(perform_logistic_hypothesis(parameters, [1, 75])))

        print('200 estimate: ' + str(perform_logistic_hypothesis(parameters, [1, 200])))

    
if __name__ == '__main__':
    main()
