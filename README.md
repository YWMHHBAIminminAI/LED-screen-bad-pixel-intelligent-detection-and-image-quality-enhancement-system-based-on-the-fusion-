# LED screen bad pixel intelligent detection and image quality enhancement system based on the fusion of traditional algorithms and deep learning
基于传统算法与深度学习融合的LED屏坏点智能检测与画质增强系统

## **技术栈**
Python, PyTorch, OpenCV, C# WPF, ONNX, Gradio, Dify/Coze

## **项目模块**

### **模块一（算法核心）**
- 使用PyTorch训练一个轻量级的坏点/缺陷检测模型（如改进的YOLO），并使用W&B记录完整的训练实验。

### **模块二（传统融合）**
- 在模型中嵌入总结的规则（如特定区域的坏点分布先验），打造“算法+知识”的混合模型。

###  **模块三（部署展示）**：
- 将模型转为ONNX，并用C# WPF封装成一个简单的上位机演示程序（体现你现有技能）。同时，用Gradio为检测模型做一个Web Demo。

### **模块四（AI赋能）**： 
- 使用**Dify**搭建一个报告生成器，上传图片即可生成带问题分析的报告。
- 使用**Coze**搭建一个问答Bot，内置你整理的画质调试知识库。

### **成果形式**：
- **Github源码库**：包含所有代码、模型权重和详细的README。
- **技术博客**：在掘金、CSDN等平台，用2-3篇文章详细阐述项目思路、技术细节和踩坑记录。
- **可访问的在线Demo**：Gradio和Coze的链接。
