#ifndef __WORKER_INTERFACE_HPP__
#define __WORKER_INTERFACE_HPP__

#include <future>
#include <memory>

#include "ThreadPool.hpp"

class WorkerInterface {
public:
	WorkerInterface() = default;

	WorkerInterface(WorkerInterface&&) = delete;
	WorkerInterface(const WorkerInterface&) = delete;

	WorkerInterface& operator=(WorkerInterface&&) = delete;
	WorkerInterface& operator=(const WorkerInterface&) = delete;

	virtual ~WorkerInterface() = default;

	virtual std::future<Matrix> AsyncProcess(Matrix) = 0;
};

class WorkerCUDA : public WorkerInterface {
public:
	std::future<Matrix> AsyncProcess(Matrix matrix) override {
		std::future<Matrix> res = thread_pool.to_work(matrix);

		return res;
	}

	WorkerCUDA(ThreadPool& thread_pool) : thread_pool(thread_pool) {

	}
private:
	ThreadPool& thread_pool;
};

class WorkerFactory {
public:
	static std::shared_ptr<WorkerInterface> create_worker() {
		return std::make_shared<WorkerCUDA>(thread_pool);
	}

private:
	static constexpr size_t thread_pool_size = 8;
	// Made decision to make 1 thread pool for all workers
	static inline ThreadPool thread_pool{thread_pool_size};
};

std::shared_ptr<WorkerInterface> get_new_worker() {
	return WorkerFactory::create_worker();
}

#endif