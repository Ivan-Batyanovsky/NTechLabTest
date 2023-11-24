#include <random>
#include <iostream>

#include "WorkerInterface.hpp"

void print_matrix(Matrix m) {
	for (size_t i = 0; i < m.height; ++i) {
		for (size_t j = 0; j < m.width; ++j) {
			std::cout << m.data[i * m.width + j] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

Matrix transpose_matrix(Matrix matrix) {
	Matrix result{std::vector<int>(matrix.height * matrix.width), matrix.height, matrix.width};
    for (size_t i = 0; i < matrix.height; i++) {
        for (size_t j = 0; j < matrix.width; j++) {
            result.data[i + j * result.width] = matrix.data[j + i * matrix.width];
		}
	}

    return result;
}

void fill_custom_tests(std::vector<Matrix> &v_m) {
	v_m.push_back({{}, 0, 0});
	v_m.push_back({{1}, 1, 1});
	v_m.push_back({{1,2}, 1, 2});
	v_m.push_back({{1,2}, 2, 1});
	v_m.push_back({{1,2,3}, 1, 3});
	v_m.push_back({{1,2,3}, 3, 1});
	v_m.push_back({{1,2,3,4}, 2, 2});
	v_m.push_back({{1,2,3,4,5,6}, 3, 2});
	v_m.push_back({{1,2,3,4,5,6}, 2, 3});
	v_m.push_back({{1,2,3,4,5,6}, 1, 6});
	v_m.push_back({{1,2,3,4,5,6}, 1, 6});
	v_m.push_back({{1,2,3,4,5,6}, 6, 1});
	v_m.push_back({{1,2,3,4,5,6,7}, 7, 1});
	v_m.push_back({{1,2,3,4,5,6,7}, 1, 7});
	v_m.push_back({{1,2,3,4,5,6,7,8}, 1, 8});
	v_m.push_back({{1,2,3,4,5,6,7,8}, 8, 1});
	v_m.push_back({{1,2,3,4,5,6,7,8}, 4, 2});
	v_m.push_back({{1,2,3,4,5,6,7,8}, 2, 4});
	v_m.push_back({{1,2,3,4,5,6,7,8,9}, 3, 3});
}

void check_results(std::vector<Matrix> &input_matrices, std::vector<std::future<Matrix>> & result_matrices) {
	if (result_matrices.size() != input_matrices.size()) {
		std::cout << "missing matrices: " << result_matrices.size() << " != " << input_matrices.size() << std::endl;
		return;
	}

	Matrix old_t_matrix, new_matrix;
	for (size_t i = 0; i < result_matrices.size(); i++)
	{
		auto new_matrix = result_matrices[i].get();
		old_t_matrix = transpose_matrix(input_matrices[i]); // imitating work of working alghoritm 
		if ((old_t_matrix.data != new_matrix.data) || (old_t_matrix.height != new_matrix.height) || (old_t_matrix.width != new_matrix.width))
		{
			std::cout << i << " :mistake found!\n";

			std::cout << old_t_matrix.width << ", " << old_t_matrix.height << std::endl;
			print_matrix(old_t_matrix);

			std::cout << new_matrix.width << ", " << new_matrix.height << std::endl;
			print_matrix(new_matrix);

			break;
		}
		
	}
}

int main() {
	auto worker = get_new_worker();
	srand((unsigned) time(NULL));

	// In addition to custom test cases I add random generated tests
	int tasks_num = 8000 + rand() % 10000;

	std::vector<Matrix> input_matrices(tasks_num);
	fill_custom_tests(input_matrices);
	std::cout << "tasks_num " << input_matrices.size() << std::endl;

	// Creating random generated matrices
	for (size_t i = 0; i < tasks_num; i++)
	{
		unsigned width = 1 + rand() % 51;
		unsigned height = 1 + rand() % 51;

		std::vector<int> data(width * height);
		for (size_t index = 0; index < width * height; index++)
			data[index] = rand() % 101;

		Matrix m{.data = data, .width = width, .height = height};
		input_matrices[i] = m;
	}
	
	std::vector<std::future<Matrix>> result_matrices;
	for (int i = 0, tasks = input_matrices.size(); i < tasks; i++) {
		result_matrices.push_back(worker->AsyncProcess(input_matrices[i]));
	}

	check_results(input_matrices, result_matrices);
}
