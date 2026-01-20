# **3. yolov5_demo.py**
"""
yolov5_demo.py - YOLOv5目标检测示例
包含: YOLOv5模型加载、推理、结果可视化
"""

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')


class YOLOv5Demo:
    """
    YOLOv5目标检测演示类
    """

    def __init__(self, model_name='yolov5s', device=None):
        """
        初始化YOLOv5模型

        Args:
            model_name: 模型名称 ('yolov5n', 'yolov5s', 'yolov5m',
                                'yolov5l', 'yolov5x')
            device: 计算设备 ('cpu', 'cuda', 'cuda:0')
        """
        print("=" * 60)
        print("YOLOv5目标检测演示")
        print("=" * 60)

        # 设置设备
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"使用设备: {self.device}")
        if self.device == 'cuda':
            print(f"GPU型号: {torch.cuda.get_device_name(0)}")

        # 加载模型
        print(f"正在加载模型: {model_name}...")
        self.model = self.load_model(model_name)

        # COCO数据集类别标签 (YOLOv5默认使用COCO)
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
            'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

        # 颜色映射 (不同类别不同颜色)
        self.colors = self.generate_colors(len(self.class_names))

        print(" ✅ 模型加载完成!")
        print(f"   类别数量: {len(self.class_names)}")
        print(f"   模型结构: {self.model.__class__.__name__}")

    def load_model(self, model_name):
        """
        加载YOLOv5模型

        注意:这里使用torch.hub加载,需要网络连接
        如果网络问题,可以使用本地模型文件
        """
        try:
            # 方法1: 使用torch.hub从官方仓库加载
            model = torch.hub.load('ultralytics/yolov5', model_name,
                                   pretrained=True)
            model.to(self.device)
            model.eval()  # 设置为评估模式

            # 打印模型信息
            print(f"   ✅ 成功加载预训练模型: {model_name}")

            return model

        except Exception as e:
            print(f"   ⚠ 无法从hub加载模型: {e}")
            print("   尝试使用本地模式...")

            # 方法2: 使用本地模型（如果有）
            # 这里可以添加本地模型加载逻辑
            raise RuntimeError("请确保网络连接正常,或提供本地模型路径")

    def generate_colors(self, n):
        """生成N种不同的颜色"""
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(n, 3), dtype=np.uint8)
        return colors

    def preprocess_image(self, image_path):
        """
        预处理图像

        Args:
            image_path: 图像路径或URL

        Returns:
            预处理后的图像张量
        """
        # 读取图像
        if isinstance(image_path, str) and image_path.startswith('http'):
            # 从URL加载
            import requests
            from PIL import Image
            import io

            response = requests.get(image_path)
            img = Image.open(io.BytesIO(response.content))
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        else:
            # 从文件加载
            img = cv2.imread(image_path)

        if img is None:
            raise ValueError(f"无法加载图像: {image_path}")

        # 保存原始图像用于显示
        self.original_img = img.copy()

        # 转换颜色空间
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img_rgb

    def detect(self, image_path, confidence_threshold=0.5):
        """
        执行目标检测

        Args:
            image_path: 图像路径或URL
            confidence_threshold: 置信度阈值

        Returns:
            检测结果 (bounding boxes, confidences, class_ids)
        """
        print(f"\n正在检测图像: {image_path}")
        print(f"置信度阈值: {confidence_threshold}")

        # 记录开始时间
        start_time = time.time()

        try:
            # 预处理图像
            img_rgb = self.preprocess_image(image_path)

            # 使用YOLOv5进行推理
            results = self.model(img_rgb)

            # 解析结果
            detections = results.pandas().xyxy[0]  # 获取检测结果

            # 过滤低置信度检测
            detections = detections[
                detections['confidence'] > confidence_threshold]

            # 计算推理时间
            inference_time = time.time() - start_time

            print("✅ 检测完成!")
            print(f"   检测到 {len(detections)} 个目标")
            print(f"   推理时间: {inference_time:.3f} 秒")
            print(f"   帧率: {1/inference_time:.1f} FPS")

            # 提取检测框信息
            boxes = []
            confidences = []
            class_ids = []

            for _, detection in detections.iterrows():
                xmin = int(detection['xmin'])
                ymin = int(detection['ymin'])
                xmax = int(detection['xmax'])
                ymax = int(detection['ymax'])
                conf = detection['confidence']
                class_id = int(detection['class'])

                boxes.append([xmin, ymin, xmax, ymax])
                confidences.append(conf)
                class_ids.append(class_id)

            return boxes, confidences, class_ids

        except Exception as e:
            print(f"❌ 检测失败: {e}")
            return [], [], []

    def draw_detections(self, boxes, confidences, class_ids):
        """
        在图像上绘制检测结果

        Args:
            boxes: 边界框列表
            confidences: 置信度列表
            class_ids: 类别ID列表

        Returns:
            绘制后的图像
        """
        img = self.original_img.copy()

        for box, conf, class_id in zip(boxes, confidences, class_ids):
            xmin, ymin, xmax, ymax = box

            # 获取类别名称和颜色
            class_name = self.class_names[class_id]
            color = self.colors[class_id].tolist()

            # 绘制边界框
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)

            # 创建标签文本
            label = f"{class_name}: {conf:.2f}"

            # 计算标签文本大小
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )

            # 绘制标签背景
            cv2.rectangle(
                img,
                (xmin, ymin - text_height - baseline - 5),
                (xmin + text_width, ymin),
                color,
                -1  # 填充
            )

            # 绘制标签文本
            cv2.putText(
                img,
                label,
                (xmin, ymin - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),  # 白色文字
                1
            )

        return img

    def visualize_results(self, img_with_boxes, boxes, confidences, class_ids):
        """
        可视化检测结果

        Args:
            img_with_boxes: 绘制了检测框的图像
            boxes: 边界框列表
            confidences: 置信度列表
            class_ids: 类别ID列表
        """
        # 创建子图
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))

        # 子图1: 原始图像
        axes[0].imshow(cv2.cvtColor(self.original_img, cv2.COLOR_BGR2RGB))
        axes[0].set_title('原始图像')
        axes[0].axis('off')

        # 子图2: 检测结果
        axes[1].imshow(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f'检测结果 ({len(boxes)}个目标)')
        axes[1].axis('off')

        # 添加统计信息
        if boxes:
            stats_text = "检测统计:\n"
            stats_text += f"• 目标数量: {len(boxes)}\n"

            # 统计每个类别的数量
            class_counts = {}
            for class_id in class_ids:
                class_name = self.class_names[class_id]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

            stats_text += "• 类别分布:\n"
            for class_name, count in class_counts.items():
                stats_text += f"  - {class_name}: {count}个\n"

            stats_text += f"• 平均置信度: {np.mean(confidences):.3f}"

            # 在图像右侧添加文本
            plt.figtext(0.75, 0.5, stats_text, fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor="lightgray"))

        plt.tight_layout()
        plt.savefig('yolov5_detection_results.png',
                    dpi=150, bbox_inches='tight')
        print("✅ 可视化结果已保存为: yolov5_detection_results.png")
        plt.show()

    def test_with_sample_image(self):
        """使用示例图像进行测试"""
        print("\n" + "=" * 60)
        print("使用示例图像测试")
        print("=" * 60)

        # 示例图像URL (来自网络)
        sample_urls = [
            "https://ultralytics.com/images/zidane.jpg",
            "https://ultralytics.com/images/bus.jpg",
        ]

        for i, url in enumerate(sample_urls):
            print(f"\n测试图像 {i+1}/{len(sample_urls)}: {url}")

            try:
                # 执行检测
                boxes, confidences, class_ids = self.detect(
                    url,
                    confidence_threshold=0.25
                )

                if boxes:
                    # 绘制检测结果
                    img_with_boxes = self.draw_detections(
                                        boxes, confidences, class_ids)

                    # 可视化
                    self.visualize_results(img_with_boxes,
                                           boxes, confidences, class_ids)

                    # 保存结果图像
                    output_path = f"yolov5_result_{i+1}.jpg"
                    cv2.imwrite(output_path, img_with_boxes)
                    print(f"✅ 结果图像已保存为: {output_path}")
                else:
                    print("⚠ 未检测到目标")

            except Exception as e:
                print(f"❌ 测试失败: {e}")

    def test_with_local_image(self, image_path):
        """使用本地图像进行测试"""
        print("\n" + "=" * 60)
        print("使用本地图像测试")
        print("=" * 60)

        if not Path(image_path).exists():
            print(f"❌ 图像文件不存在: {image_path}")
            print("请提供本地图像路径或使用示例图像")
            return

        try:
            # 执行检测
            boxes, confidences, class_ids = self.detect(
                image_path,
                confidence_threshold=0.25
            )

            if boxes:
                # 绘制检测结果
                img_with_boxes = self.draw_detections(boxes,
                                                      confidences, class_ids)

                # 可视化
                self.visualize_results(img_with_boxes,
                                       boxes, confidences, class_ids)

                # 保存结果图像
                output_path = "yolov5_local_result.jpg"
                cv2.imwrite(output_path, img_with_boxes)
                print(f"✅ 结果图像已保存为: {output_path}")
            else:
                print("⚠ 未检测到目标")

        except Exception as e:
            print(f"❌ 测试失败: {e}")

    def benchmark_performance(self):
        """性能基准测试"""
        print("\n" + "=" * 60)
        print("性能基准测试")
        print("=" * 60)

        # 创建测试图像
        test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        cv2.imwrite("test_benchmark.jpg", test_img)

        # 预热
        print("预热模型...")
        for _ in range(3):
            _ = self.detect("test_benchmark.jpg", confidence_threshold=0.5)

        # 正式测试
        n_tests = 10
        print(f"进行 {n_tests} 次推理测试...")

        inference_times = []

        for i in range(n_tests):
            start_time = time.time()
            boxes, _, _ = self.detect("test_benchmark.jpg",
                                      confidence_threshold=0.5)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)

            print(f"  测试 {i+1}/{n_tests}: {inference_time:.3f}s, "
                  f"  检测到 {len(boxes)} 个目标")

        # 清理测试文件
        Path("test_benchmark.jpg").unlink(missing_ok=True)

        # 计算统计信息
        avg_time = np.mean(inference_times)
        std_time = np.std(inference_times)
        fps = 1 / avg_time

        print("\n📊 性能测试结果:")
        print(f"  平均推理时间: {avg_time:.3f} ± {std_time:.3f} 秒")
        print(f"  平均帧率: {fps:.1f} FPS")
        print(f"  最快推理: {min(inference_times):.3f} 秒 "
              f"  ({1/min(inference_times):.1f} FPS)")
        print(f"  最慢推理: {max(inference_times):.3f} 秒 "
              f"  ({1/max(inference_times):.1f} FPS)")

        # 可视化性能结果
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.plot(range(1, n_tests + 1), inference_times,
                 'bo-', linewidth=2, markersize=8)
        plt.axhline(y=avg_time, color='r', linestyle='--',
                    label=f'平均: {avg_time:.3f}s')
        plt.xlabel('测试次数')
        plt.ylabel('推理时间 (秒)')
        plt.title('YOLOv5推理时间')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        fps_values = [1/t for t in inference_times]
        plt.bar(range(1, n_tests + 1), fps_values, color='green', alpha=0.7)
        plt.axhline(y=fps, color='r', linestyle='--',
                    label=f'平均: {fps:.1f} FPS')
        plt.xlabel('测试次数')
        plt.ylabel('帧率 (FPS)')
        plt.title('YOLOv5推理帧率')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('yolov5_performance_benchmark.png',
                    dpi=150, bbox_inches='tight')
        print("✅ 性能图表已保存为: yolov5_performance_benchmark.png")
        plt.show()


def main():
    """主函数"""
    print("YOLOv5目标检测演示")
    print("=" * 60)

    try:
        # 1. 初始化YOLOv5模型
        print("1. 初始化模型...")
        detector = YOLOv5Demo(model_name='yolov5s')

        # 2. 测试选项
        print("\n2. 选择测试模式:")
        print("   1. 使用示例图像测试 (需要网络)")
        print("   2. 使用本地图像测试")
        print("   3. 性能基准测试")

        choice = input("\n请选择 (1-3): ").strip()

        if choice == '1':
            # 使用示例图像测试
            detector.test_with_sample_image()

        elif choice == '2':
            # 使用本地图像测试
            image_path = input("请输入本地图像路径: ").strip()
            detector.test_with_local_image(image_path)

        elif choice == '3':
            # 性能基准测试
            detector.benchmark_performance()

        else:
            print("⚠ 无效选择,使用示例图像测试")
            detector.test_with_sample_image()

        print("\n" + "=" * 60)
        print("🎉 YOLOv5演示完成!")
        print("=" * 60)

        # 打印模型信息
        print("\n📋 模型信息:")
        print("   模型: yolov5s")
        print("   参数量: 约7百万")
        print("   输入尺寸: 640x640")
        print("   类别数: 80 (COCO数据集)")

    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

        print("\n💡 故障排除建议:")
        print("   1. 检查网络连接 (需要下载预训练模型)")
        print("   2. 安装依赖: pip install opencv-python matplotlib")
        print("   3. 确保PyTorch已正确安装")
        print("   4. 如仍有问题,可尝试其他模型或本地模型")


if __name__ == "__main__":
    main()
