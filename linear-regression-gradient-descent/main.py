LEARNING_RATE = 0.0001

######TODO: This is doing summing wrong
##########  each gradient descent step should sum over all training samples
##########  - also need to repeat until convergance


def cost_function():
    print(1)

def gradient_descent_step_single_parameter(parameters, data_point, label, parameter_index):
    parameter = parameters[parameter_index]
    data_point_feature = data_point[parameter_index]

    new_parameter = parameter - LEARNING_RATE * (perform_linear_hypothesis(parameters, data_point) - label) * data_point_feature

    parameters[parameter_index] = new_parameter

    return parameters


def gradient_descent_step(parameters, data_point, label):
    for i in range(len(parameters)):
        parameters = gradient_descent_step_single_parameter(parameters, data_point, label, i)

    return parameters

def perform_linear_hypothesis(parameters, data_point):
    result = 0

    for i in range(len(parameters)):
        result += parameters[i] * data_point[i]
    return result


def main():
    #n: number of features
    #m: number of training data points

    n = 3
    m = 7
    

    labels = range(1,100) # array size m
    training_data_set = [
        [1, 1, 234, 999],
        [1, 2, 532, 38],
        [1, 3, 342, 0],
        [1, 4, 635, 423],
        [1, 5, 9352, 84834],
        [1, 6, 8385, 8294],
        [1, 7, 8385, 2432],
        [1, 8, 3853, 8305],
        [1, 9, 4859, 8352],
        [1, 10, 4852, 6840],
        [1, 11, 234, 999],
        [1, 12, 532, 38],
        [1, 13, 342, 0],
        [1, 14, 635, 423],
        [1, 15, 9352, 84834],
        [1, 16, 8385, 8294],
        [1, 17, 8385, 2432],
        [1, 18, 3853, 8305],
        [1, 19, 4859, 8352],
        [1, 20, 4852, 6840],
        [1, 21, 234, 999],
        [1, 22, 532, 38],
        [1, 23, 342, 0],
        [1, 24, 635, 423],
        [1, 25, 9352, 84834],
        [1, 26, 8385, 8294],
        [1, 27, 8385, 2432],
        [1, 28, 3853, 8305],
        [1, 29, 4859, 8352],
        [1, 30, 4852, 6840],
        [1, 31, 234, 999],
        [1, 32, 532, 38],
        [1, 33, 342, 0],
        [1, 34, 635, 423],
        [1, 35, 9352, 84834],
        [1, 36, 8385, 8294],
        [1, 37, 8385, 2432],
        [1, 38, 3853, 8305],
        [1, 39, 4859, 8352],
        [1, 40, 4852, 6840],
        [1, 41, 234, 999],
        [1, 42, 532, 38],
        [1, 43, 342, 0],
        [1, 44, 635, 423],
        [1, 45, 9352, 84834],
        [1, 46, 8385, 8294],
        [1, 47, 8385, 2432],
        [1, 48, 3853, 8305],
        [1, 49, 4859, 8352],
        [1, 50, 4852, 6840],
        [1, 51, 234, 999],
        [1, 52, 532, 38],
        [1, 53, 342, 0],
        [1, 54, 635, 423],
        [1, 55, 9352, 84834],
        [1, 56, 8385, 8294],
        [1, 57, 8385, 2432],
        [1, 58, 3853, 8305],
        [1, 59, 4859, 8352],
        [1, 60, 4852, 6840],
        [1, 61, 234, 999],
        [1, 62, 532, 38],
        [1, 63, 342, 0],
        [1, 64, 635, 423],
        [1, 65, 9352, 84834],
        [1, 66, 8385, 8294],
        [1, 67, 8385, 2432],
        [1, 68, 3853, 8305],
        [1, 69, 4859, 8352],
        [1, 70, 4852, 6840],
        [1, 71, 234, 999],
        [1, 72, 532, 38],
        [1, 73, 342, 0],
        [1, 74, 635, 423],
        [1, 75, 9352, 84834],
        [1, 76, 8385, 8294],
        [1, 77, 8385, 2432],
        [1, 78, 3853, 8305],
        [1, 79, 4859, 8352],
        [1, 80, 4852, 6840],
        [1, 81, 234, 999],
        [1, 82, 532, 38],
        [1, 83, 342, 0],
        [1, 84, 635, 423],
        [1, 85, 9352, 84834],
        [1, 86, 8385, 8294],
        [1, 87, 8385, 2432],
        [1, 88, 3853, 8305],
        [1, 89, 4859, 8352],
        [1, 90, 4852, 6840],
        [1, 91, 234, 999],
        [1, 92, 532, 38],
        [1, 93, 342, 0],
        [1, 94, 635, 423],
        [1, 95, 9352, 84834],
        [1, 96, 8385, 8294],
        [1, 97, 8385, 2432],
        [1, 98, 3853, 8305],
        [1, 99, 4859, 8352],
        [1, 100, 4852, 6840],
    ] # 2d array size m x (n + 1)


    parameters = [1, 1, 1, 1] # array size (n + 1)

    for i in range(len(training_data_set)):
        data_point = training_data_set[i]
        label = labels[i]

        parameters = gradient_descent_step(parameters, data_point, label)
        print(f'Parameters: {parameters}')
        

    
if __name__ == '__main__':
    main()
