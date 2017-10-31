#include "test.h"

int pool = 0;
std::mutex poolMutex;
std::condition_variable cv;

void selfInc()
{
	while (true)
	{
		
		{
			std::unique_lock<std::mutex> lck(poolMutex);
			pool++;
			if (pool >= 10)
			{
				std::cout << "up to 10" << "   try to stop!" << std::endl;
				cv.notify_all();
				break;
			}
			std::cout << "the pool is " << pool << std::endl;
		}
		
		std::this_thread::sleep_for(std::chrono::seconds(1));
	}
}

void detect()
{
	
	std::unique_lock<std::mutex>lck(poolMutex);
	std::cout << "begin to check......" << std::endl;
	while (pool < 10)
	{
		cv.wait(lck);
		std::cout << "hello! beging to stop!" << std::endl;
	}
}


//int main()
//{
//	int input = 0;
//	std::cin >> input;
//	std::shared_ptr < Status > ptr= getStatus(input);
//	ptr->process();
//	return 0;
//}