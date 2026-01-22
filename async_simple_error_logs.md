async_simple_error_logs_20260121


block comment should start with '# '     # 注释符后要加空格


missing whitespace after ','   # 逗号冒号后面要加空格 


at least two spaces before inline comment   # 句子后隔两个空格再注释


line too long (91 > 79 characters)
   # 算式超长，等号右边可用括号把内容单独放一行
   # if 语句超长，新建变量缩短语句
   # 网址超长，分两段用 ' ' 接上，上下两段要对齐



local variable 'async_results' is assigned to but never used
   # 以下代码把结果收进了 async_results，但后面一句都没用它，静态检查器就会报 “assigned but never used”。如果只是想消除警告，又不准备打印结果，就把返回值丢掉，用占位符 _：
_ = await asyncio.gather(*tasks)   # 结果不需要
   # 但考虑到加了这句进任务里相当于跑两次，会造成同步异步速度对比结果偏差，所以还是要只跑一次，把返回值赋给 _，或者干脆把变量名也省掉：
await asyncio.gather(*[async_task(i, d) for i, d in enumerate(delays)])
   # 报错代码：
    async def async_task(task_id: int, delay: float):
        """异步任务"""
        await asyncio.sleep(delay)
        return f"异步任务 {task_id}"

    start_time = time.time()
    tasks = [async_task(i, delay) for i, delay in enumerate(delays)]
    async_results = await asyncio.gather(*tasks)
    async_time = time.time() - start_time

    print(f" 用时: {async_time:.2f} 秒")
    print(f" 平均每个任务: {async_time/len(delays):.2f} 秒")



使用asyncio.wait:
❌ 错误: Passing coroutines is forbidden, use tasks explicitly.
Traceback (most recent call last):
  File "f:\AI_prac\async_simple.py", line 587, in main
    await basic_async_examples()
  File "f:\AI_prac\async_simple.py", line 280, in basic_async_examples
    done, pending = await asyncio.wait(tasks, timeout=1.0)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\miniconda3\envs\ai-deploy311\Lib\asyncio\tasks.py", line 425, in wait
    raise TypeError("Passing coroutines is forbidden, use tasks explicitly.")
TypeError: Passing coroutines is forbidden, use tasks explicitly.
f:\AI_prac\async_simple.py:619: RuntimeWarning: coroutine 'basic_async_examples.<locals>.task_with_id' was never awaited
  traceback.print_exc()
RuntimeWarning: Enable tracemalloc to get the object allocation traceback
   # asyncio.wait 只接受 Task 对象，不能直接给协程；把列表推导换成 create_task 即可。
   # 旧代码（报错）
tasks = [task_with_id(i, 0.5 + i*0.1) for i in range(5)]
done, pending = await asyncio.wait(tasks, timeout=1.0)
   # 修正
tasks = [asyncio.create_task(task_with_id(i, 0.5 + i*0.1)) for i in range(5)]
done, pending = await asyncio.wait(tasks, timeout=1.0)


下载全部失败——SSL + 超时
   #原因：
· via.placeholder.com 证书主机名不匹配；
· 国内网络经常连不上。
   #修复：直接关 SSL 校验 + 缩短超时，先让 Demo 能跑。
在 AsyncDownloader.download_file 里改一行即可：
   # 旧
async with session.get(url) as response:
   # 新
async with session.get(url, ssl=False, timeout=aiohttp.ClientTimeout(total=10)) as response:
（如介意关 SSL，可把 ssl=False 换成 ssl=aiohttp.Fingerprint(...) 或自己挂代理。）


同步计时爆掉——176 亿秒
   原因：sync_time = time.time() 没减去 start_time，导致把 Unix 时间戳当成耗时。
   修：sync_time = time.time() - start_time


Matplotlib 炸掉——巨图 + 中文缺字
   原因：把 speedup 当成字符串 'speedup' 传进 bar()，结果画出来的图宽 1514 px、高 11522080231046 px，直接溢出；
中文字体缺失会报警告，但程序仍能跑，先忽略。
   修复：
①'speed' 去掉引号
②添加中文，在文件开头
# -*- coding: utf-8 -*-
import matplotlib as mpl
# 选用系统自带雅黑
mpl.rcParams['font.family'] = 'Microsoft YaHei'
mpl.rcParams['axes.unicode_minus'] = False


运行4.高级模式会卡住如下
   #卡住进度：
1. 异步队列(生产者-消费者模式):
 生产者: 生产了 产品0
...
 生产者: 生产了 产品9
 消费者2: 消费了 产品9
#错误代码：
 async def producer(queue: asyncio.Queue, n: int):
        """生产者"""
        for i in range(n):
            item = f"产品{i}"
            await queue.put(item)
            print(f" 生产者: 生产了 {item}")
            await asyncio.sleep(0.1)
        await queue.put(None)  # 结束信号

    async def consumer(queue: asyncio.Queue, consumer_id: int):
        """消费者"""
        while True:
            item = await queue.get()
            if item is None:
                queue.put_nowait(None)  # 传递结束信号
                break
            print(f" 消费者{consumer_id}: 消费了 {item}")
            await asyncio.sleep(0.2)  # 模拟处理时间
            queue.task_done()
   #原因：queue.put(None) 只发了一次结束令牌，却在后面被几个消费者抢；只要有一个消费者先拿到 None 就 break，其他两个永远等不到下一个 None，于是 queue.join() 永远不满足，整个协程就悬在那里。
   #修复：给每个消费者都发一枚“退出币”
      # 生产后发 N 枚退出币
    for _ in range(N_CONSUMERS):
        await queue.put(None)
      #消费者拿到币就退出，不回传
item = await queue.get()
        if item is None:          # 拿到退出币
            queue.task_done()
            break                 # 直接走人，不再回传

