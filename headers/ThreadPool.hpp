#ifndef __THREAD_POOL_HPP__
#define __THREAD_POOL_HPP__

#include <vector>
#include <future>
#include <queue>
#include <functional>
#include <mutex>
#include <utility>
#include <thread>

#include "CudaFuncs.hpp"
#include "Matrix.hpp"

#include <driver_types.h>
#include <cuda_runtime_api.h>

class ThreadPool
{
public:

	/// @brief Used to put matrix transposition to tasks queue 
	/// @param matrix matrix to transpose
	/// @return future of matrix which is processed asynchronously
	std::future<Matrix> to_work(Matrix & matrix);

	ThreadPool(size_t thread_pool_size);
	~ThreadPool();

private:
	std::vector<std::thread> threads;
	std::queue<std::packaged_task<Matrix(cudaStream_t)>> tasks;

	std::mutex tasks_mtx;
	std::condition_variable conditional;
	
	bool stop_called = false;
};

inline std::future<Matrix> ThreadPool::to_work(Matrix & matrix)
{
	auto task = std::packaged_task<Matrix(cudaStream_t)>(std::bind(cudaWrapper, matrix, std::placeholders::_1));

	auto result = task.get_future();
	
	{
		std::unique_lock<std::mutex> ul(tasks_mtx);
		tasks.emplace(std::move(task));
	}
	
	conditional.notify_one();

	return result;
}

inline ThreadPool::ThreadPool(size_t thread_pool_size)
{
	for (size_t i = 0; i < thread_pool_size; i++)
	{
		threads.emplace_back([this] {
			std::packaged_task<Matrix(cudaStream_t)> task;

			// CUDA stream for async function calls 
			cudaStream_t stream;
			cudaStreamCreate(&stream);

			while (true)
			{
				{
					std::unique_lock<std::mutex> ul(tasks_mtx);
					conditional.wait(ul, [this]{ return stop_called || !tasks.empty();});
					if (stop_called && tasks.empty()) { // threads completing tasks deliberately even if stop is called
						cudaStreamDestroy(stream);
						return;
					}

					task = std::move(tasks.front());
					tasks.pop();
				}
				task(stream);
			}
		});
	}
}

inline ThreadPool::~ThreadPool()
{
	{
		std::unique_lock<std::mutex> ul(tasks_mtx);
		stop_called = true;
	}

	conditional.notify_all();

	for (auto & thread : threads) {
		thread.join();
	}
}

#endif