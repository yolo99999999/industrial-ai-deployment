# -*- coding: utf-8 -*-

"""
async_simple.py - 异步编程基础
包含: async/await基础、并发下载、任务管理
"""


import asyncio
import aiohttp
import time
import os
from pathlib import Path
from typing import List, Dict
import hashlib
import json
import matplotlib as mpl
# 选用系统自带雅黑
mpl.rcParams['font.family'] = 'Microsoft YaHei'
mpl.rcParams['axes.unicode_minus'] = False


class AsyncDownloader:
    """异步下载器类"""

    def __init__(self, max_concurrent=5):
        """
        初始化异步下载器

        Args:
            max_concurrent: 最大并发数
        """
        self.max_concurrent = max_concurrent
        self.results = []
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'total_size': 0,
            'total_time': 0
        }

    async def download_file(self, session: aiohttp.ClientSession,
                            url: str, save_path: str) -> Dict:
        """
        异步下载单个文件

        Args:
            session: aiohttp会话
            url: 下载URL
            save_path: 保存路径

        Returns:
            下载结果字典
        """
        start_time = time.time()
        result = {
            'url': url,
            'save_path': save_path,
            'success': False,
            'error': None,
            'size': 0,
            'time': 0,
            'checksum': None
        }

        try:
            # 发起异步请求
            async with session.get(
                    url, ssl=False, timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    # 读取内容
                    content = await response.read()

                    # 计算文件大小和校验和
                    file_size = len(content)
                    checksum = hashlib.md5(content).hexdigest()

                    # 保存文件
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    with open(save_path, 'wb') as f:
                        f.write(content)

                    # 更新结果
                    result.update({
                        'success': True,
                        'size': file_size,
                        'time': time.time() - start_time,
                        'checksum': checksum,
                        'status': response.status
                    })

                    print(f"✅ 下载成功： { url}")
                    print(f" 保存到: {save_path}")
                    print(f" 大小: {file_size} 字节")
                    print(f" 用时: {result['time']:.2f} 秒")

                else:
                    result.update({
                        'error': "HTTP错误: {response.status}",
                        'status': response.status
                    })
                    print(f"❌ 下载失败: {url} - HTTP {response.status}")

        except Exception as e:
            result.update({
                'error': str(e),
                'time': time.time() - start_time
            })
            print(f"❌ 下载失败: {url} - {e}")

        return result

    async def download_with_semaphore(self, session: aiohttp.ClientSession,
                                      semaphore: asyncio.Semaphore,
                                      url: str, save_path: str) -> Dict:
        """
        使用信号量控制并发下载

        Args:
            session: aiohttp会话
            semaphore: 信号量
            url: 下载URL
            save_path: 保存路径

        Returns:
            下载结果字典
        """
        async with semaphore:
            return await self.download_file(session, url, save_path)

    async def download_multiple(
            self, url_list: List[Dict[str, str]]) -> List[Dict]:
        """
        异步下载多个文件

        Args:
            url_list: URL列表,每个元素包含'url和'save_path'

        Returns:
            下载结果列表
        """
        print("="*60)
        print("开始异步下载任务")
        print("="*60)
        print(f" 总任务数: {len(url_list)}")
        print(f" 最大并发数: {self.max_concurrent}")
        print(f" 开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)

        start_time = time.time()

        #  创建信号量控制并发数
        semaphore = asyncio.Semaphore(self.max_concurrent)

        # 创建aiohttp会话
        async with aiohttp.ClientSession() as session:
            # 创建所有下载任务
            tasks = []
            for item in url_list:
                task = self.download_with_semaphore(
                    session, semaphore,
                    item['url'], item['save_path']
                )
                tasks.append(task)

            # 等待所有任务完成
            results = await asyncio.gather(*tasks)

        # 计算统计信息
        total_time = time.time() - start_time

        self.results = results
        self.stats.update({
            'total': len(results),
            'success': sum(1 for r in results if r['success']),
            'failed': sum(1 for r in results if not r['success']),
            'total_size': sum(r['size'] for r in results if r['success']),
            'total_time': total_time
        })

        return results

    def print_statistics(self):
        """打印统计信息"""
        print("\n"+"="*60)
        print("下载统计")
        print("="*60)

        stats = self.stats

        print(f" 总任务数: {stats['total']}")
        print(f" 成功: {stats['success']}")
        print(f" 失败: {stats['failed']}")
        print(f" 成功率: {stats['success']/stats['total']*100:.1f}%")
        print(f" 总下载大小: {stats['total_size']:,} 字节"
              f" ({stats['total_size']/1024/1024:.2f} MB)")
        print(f" 总用时: {stats['total_time']:.2f} 秒")

        if stats['success'] > 0:
            avg_speed = (
                stats['total_size'] / stats['total_time'] / 1024  # KB/S
            )
            print(f" 平均速度: {avg_speed:.2f} 秒")

        # 显示失败的URL
        failed_urls = [r['url'] for r in self.results if not r['success']]
        if failed_urls:
            print("\n失败的URL:")
            for url in failed_urls:
                print(f" - {url}")

        # 保存结果到文件
        self.save_results()

    def save_results(self):
        """保存下载结果到文件"""
        #  准备可JSON序列化的结果
        serializable_results = []
        for result in self.results:
            serializable_result = result.copy()
            # 移除可能不可序列化的字段
            err = serializable_result.get('error')
            if err is not None:
                serializable_result['error'] = str(err)
            serializable_results.append(serializable_result)

        # 保存为JSON
        with open('download-results.json', 'w', encoding='utf-8') as f:
            json.dump({
                'stats': self.stats,
                'results': serializable_results,
                'timestamp': time.strftime('%Y-%m-%d %H:5M:%S')
            }, f, indent=2, ensure_ascii=False)

        print("✅ 结果已保存到: download_results.json")


async def basic_async_examples():
    """基础异步编程示例"""
    print("="*60)
    print("基础异步编程示例")
    print("="*60)

    # 示例1: 简单的异步函数
    print("\n1. 简单异步函数")

    async def say_hello(name: str, delay: float):
        """异步打招呼函数"""
        print(f" [开始] 向 {name} 打招呼")
        await asyncio.sleep(delay)  # 模拟耗时操作
        print(f" [完成] 你好, {name}!")
        return f"hello {name}"

    # 创建并运行异步任务
    task1 = say_hello("Alice", 1.0)
    task2 = say_hello("Bob", 0.5)
    task3 = say_hello("Charlie", 0.3)

    results = await asyncio.gather(task1, task2, task3)
    print(f" 所有任务完成, 结果: {results}")

    # 示例2: 使用async for循环
    print("\n2. 异步生成器")

    async def async_counter(n: int):
        """异步计数器"""
        for i in range(n):
            yield i
            await asyncio.sleep(0.1)  # 模拟异步操作

    async for number in async_counter(5):
        print(f" 计数器: {number}")

    # 示例3: 使用asyncio.wait
    print(" \n3. 使用asyncio.wait:")

    async def task_with_id(task_id: int, delay: float):
        """带ID的任务"""
        await asyncio.sleep(delay)
        return f"任务完成{task_id}完成"

    # 创建多个任务
    tasks = [
        asyncio.create_task(task_with_id(i, 0.5 + i*0.1)) for i in range(5)
        ]

    # 等待所有任务完成, 设置超时
    done, pending = await asyncio.wait(tasks, timeout=1.0)

    print(f" 已完成: {len(done)} 个任务")
    print(f" 未完成: {len(pending)} 个任务")

    # 获取已完成任务的结果
    for task in done:
        print(f" {task.result()}")

    # 取消未完成的任务
    for task in pending:
        task.cancel()

    # 示例4: 异步上下文管理器
    print("\n4. 异步上下文管理器:")

    class AsyncResource:
        """模拟异步资源"""
        async def __aenter__(self):
            print(" 正在获取资源...")
            await asyncio.sleep(0.2)
            print(" 资源已获取")
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            print(" 正在释放资源...")
            await asyncio.sleep(0.1)
            print(" 资源已释放")

        async def process(self):
            """处理资源"""
            print(" 正在处理资源")
            await asyncio.sleep(0.3)
            print(" 资源处理完成")

    async with AsyncResource() as resource:
        await resource.process()

    print("\n✅ 基础示例完成")


async def download_example_images():
    """下载示例图像"""
    print("\n"+"="*60)
    print("异步下载示例图像")
    print("="*60)

    # 准备要下载的图像URL列表
    image_urls = [
        {'url': 'https://picsum.photos/400/300?random=1',
         'save_path': 'downloads/img1.jpg'},
        {'url': 'https://picsum.photos/400/300?random=2',
         'save_path': 'downloads/img2.jpg'},
        {'url': 'https://picsum.photos/400/300?random=3',
         'save_path': 'downloads/img3.jpg'},
        {'url': 'https://picsum.photos/400/300?random=4',
         'save_path': 'downloads/img4.jpg'},
        {'url': 'https://picsum.photos/400/300?random=5',
         'save_path': 'downloads/img5.jpg'},
        {'url': 'https://picsum.photos/400/300?random=6',
         'save_path': 'downloads/img6.jpg'},
        {'url': 'https://picsum.photos/400/300?random=7',
         'save_path': 'downloads/img7.jpg'},
        {'url': 'https://picsum.photos/400/300?random=8',
         'save_path': 'downloads/img8.jpg'},
    ]

    # 创建下载器实例
    downloader = AsyncDownloader(max_concurrent=3)

    # 执行下载
    await downloader.download_multiple(image_urls)

    # 打印统计信息
    downloader.print_statistics()

    # 显示下载的文件
    print("\n📁 下载的文件:")
    download_dir = Path("downloads")
    if download_dir.exists():
        for file_path in download_dir.glob("*.jpg"):
            file_size = file_path.stat().st_size
            print(f" - {file_path.name} {file_size:,} 字节")


async def compare_sync_vs_async():
    """对比同步和异步性能"""
    print("\n" + "="*60)
    print("同步 vs 异步性能对比")
    print("="*60)

    # 模拟的网络请求延迟
    delays = [0.5, 0.3, 0.8, 0.2, 0.4, 0.6, 0.1, 0.7]

    # 同步版本
    print("1. 同步版本")

    def sync_task(task_id: int, delay: float):
        """同步任务"""
        time.sleep(delay)
        return f"同步任务{task_id}"

    start_time = time.time()
    sync_results = []
    for i, delay in enumerate(delays):
        result = sync_task(i, delay)
        sync_results.append(result)
    sync_time = time.time() - start_time

    print(f" 用时: {sync_time:.2f} 秒")
    print(f" 平均每个任务: {sync_time/len(delays):.2f} 秒")

    # 异步版本
    print("\n2. 异步版本")

    async def async_task(task_id: int, delay: float):
        """异步任务"""
        await asyncio.sleep(delay)
        return f"异步任务 {task_id}"

    start_time = time.time()
    await asyncio.gather(*[async_task(i, d) for i, d in enumerate(delays)])
    async_time = time.time() - start_time

    print(f" 用时: {async_time:.2f} 秒")
    print(f" 平均每个任务: {async_time/len(delays):.2f} 秒")

    # 对比结果
    print("\n📊 性能对比")
    print(f" 同步总时间: {sync_time:.2f} 秒")
    print(f" 异步总时间: {async_time:.2f} 秒")
    print(f" 加速比: {sync_time/async_time:.2f}x")
    print(f" 时间节省: {(sync_time - async_time):.2f} 秒")

    # 可视化对比
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 子图1: 时间对比
    methods = ['同步', '异步']
    times = [sync_time, async_time]

    axes[0].bar(methods, times, color=['red', 'green'])
    axes[0].set_ylabel('时间(秒)')
    axes[0].set_title('同步 vs 异步执行时间')
    axes[0].grid(True, alpha=0.3)

    # 添加数值标签
    for i, v in enumerate(times):
        axes[0].text(i, v+0.1, f'{v:.2f}s', ha='center')

    # 子图2: 加速比
    speedup = sync_time / async_time
    axes[1].bar(['加速比'], [speedup], color=['blue'])
    axes[1].set_ylabel('倍数')
    axes[1].set_title('异步加速比: {speedup:.2f}x')
    axes[1].grid(True, alpha=0.3)
    axes[1].text(0, speedup + 0.1, f'{speedup:.2f}x', ha='center')

    plt.tight_layout()
    plt.savefig('sync_vs_async_comparison.png', dpi=150, bbox_inches='tight')
    print("✅ 对比图表已保存为: sync_vs_async_comparison.png")
    plt.show()


async def advanced_async_patterns():
    """高级异步模式"""
    print("\n" + "="*60)
    print("同步 vs 异步性能对比")
    print("="*60)

    # 模式1: 异步队列
    print("\n1. 异步队列(生产者-消费者模式):")

    n_consumers = 4

    async def producer(queue: asyncio.Queue, n: int):
        """生产者"""
        for i in range(n):
            item = f"产品{i}"
            await queue.put(item)
            print(f" 生产者: 生产了 {item}")
            await asyncio.sleep(0.1)
        # 发 N 枚退出币
        for _ in range(n_consumers):
            await queue.put(None)  # 结束信号

    async def consumer(queue: asyncio.Queue, consumer_id: int):
        """消费者"""
        while True:
            item = await queue.get()
            if item is None:  # 传递结束信号
                queue.task_done()
                break  # 直接退出
            print(f" 消费者{consumer_id}: 消费了 {item}")
            await asyncio.sleep(0.2)  # 模拟处理时间
            queue.task_done()

    # 创建队列和任务
    queue = asyncio.Queue(maxsize=5)

    producer_task = asyncio.create_task(producer(queue, 10))
    consumer_tasks = [
        asyncio.create_task(consumer(queue, i)) for i in range(n_consumers)]

    # 等待完成
    await producer_task
    await queue.join()

    for task in consumer_tasks:
        task.cancel()

    # 模式2: 异步锁
    print("\n2. 异步锁(保护共享资源):")

    shared_counter = 0
    lock = asyncio.Lock()

    async def increment_counter(task_id: int):
        """递增计数器"""
        nonlocal shared_counter

        async with lock:  # 使用锁保护临界区
            print(f" 任务{task_id}: 获取锁")
            await asyncio.sleep(0.1)  # 模拟耗时操作
            shared_counter += 1
            print(f" 任务{task_id}: 计数器 = {shared_counter}")

    # 创建多个并发递增任务
    tasks = [increment_counter(i) for i in range(5)]
    await asyncio.gather(*tasks)

    print(f" 最终计数器值: {shared_counter}")

    # 模式3: 异步事件
    print("\n3. 异步事件(协调多个任务):")

    event = asyncio.Event()

    async def waiter(task_id: int):
        """"等待事件"""
        print(f" 等待者{task_id}: 等待事件...")
        await event.wait()
        print(f" 等待着{task_id}: 事件已触发!")

    async def trigger():
        """触发事件"""
        print(" 触发器: 等待3秒后触发事件...")
        await asyncio.sleep(3)
        event.set()
        print(" 触发器: 事件已触发!")

    # 创建任务
    waiter_tasks = [asyncio.create_task(waiter(i)) for i in range(3)]
    trigger_task = asyncio.create_task(trigger())

    await asyncio.gather(*waiter_tasks, trigger_task)

    print("\n✅ 高级模式演示完成!")


async def main():
    """主函数"""
    print("异步编程基础演示")
    print("="*60)

    try:
        # 演示选项
        print("\n请选择演示内容:")
        print("1. 基础异步示例")
        print("2. 异步下载示例")
        print("3. 同步vs异步性能对比")
        print("4. 高级异步模式")
        print("5. 全部演示")

        choice = input("\n请选择(1-5): ").strip()

        if choice == '1':
            await basic_async_examples()
        elif choice == '2':
            await download_example_images()
        elif choice == '3':
            await compare_sync_vs_async()
        elif choice == '4':
            await advanced_async_patterns()
        elif choice == '5':
            await basic_async_examples()
            await download_example_images()
            await compare_sync_vs_async()
            await advanced_async_patterns()
        else:
            print("⚠ 无法选择, 执行基础示例")
            await basic_async_examples()

        print("\n" + "="*60)
        print("🎉 异步编程演示完成!")
        print("="*60)

        # 总结
        print("\n💡 异步编程关键点:")
        print(" 1.async/await: 定义和调用异步函数")
        print(" 2.asyncio.run(): 运行异步程序")
        print(" 3.asyncio.gather(): 并发执行多个任务")
        print(" 4.asyncio.create_task(): 创建后台任务")
        print(" 5.使用信号量控制并发数")
        print(" 6.异步上下文管理器(async with)")
        print(" 7.异步迭代器(async for)")

        print("\n📚 应用场景:")
        print(" · 网络请求 (HTTP/WebSocket)")
        print(" · 文件I/O(异步读写)")
        print(" · 数据库操作")
        print(" · Web服务器(FastAPI/Starlette)")
        print(" · 实时数据处理")

    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())
