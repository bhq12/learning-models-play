import numpy

def transpose(matrix: list[list]): 
    if len(matrix) == 0:
        return matrix
    row_count = len(matrix) 
    column_count = len(matrix[0])
    transposed_matrix = []
    for i in range(column_count):
        new_row = []

        for j in range(row_count):
            new_row.append(matrix[j][i])
        transposed_matrix.append(new_row)
    return transposed_matrix
#NOTE: None of this is very efficient from a data-locatity standpoint
# but gets the job done for small expamples
def matrix_multiply(matrix_1: list[list], matrix_2: list[list]):
    if len(matrix_1) == 0 or len(matrix_2) == 0:
        return None
    
    if len(matrix_1[0]) != len(matrix_2):
        return -1

    result = []

    result_rows = len(matrix_1)

    result_columns = len(matrix_2[0])

    for row_index in range(result_rows):
        result.append([])
        matrix_1_row = matrix_1[row_index]
        for column_index in range(result_columns):
            result_value = 0

            for multiplication_index in range(len(matrix_1_row)):
                multiplication = matrix_1_row[multiplication_index] * matrix_2[multiplication_index][column_index]
                result_value += multiplication

            result[row_index].append(result_value)
    return result
            
def matrix_inverse(matrix: list):
    # Cheating but I'm trying to do machine learning not re-invent linear algebra
    data_frame = numpy.array(matrix)

    return numpy.linalg.inv(data_frame).tolist()

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
        

    
if __name__ == '__main__':
    # main()
    matrix = [[1, 2,3],[4,5,6]]

    transposed = transpose(matrix)

    print(matrix)
    print(transposed)

    multiplied = matrix_multiply(matrix, transposed) 
    print('multiplied')
    print(multiplied)
    inverse = matrix_inverse(multiplied)
    print('inverse')
    print(inverse)
    # Expect identity matrix 
    print('inverse multiplied')
    print(matrix_multiply(multiplied, inverse))
