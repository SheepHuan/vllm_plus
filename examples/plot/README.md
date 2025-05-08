# TTFT vs 准确率 绘图工具

这个脚本用于绘制不同模型在各种数据集上的"首词生成时间(TTFT)"与"准确率(F1-score/Rouge-L-score)"之间的关系图。

## 功能

- 绘制多模型、多数据集的对比图表
- 支持四种不同的缓存方法比较
- 可自定义输入数据和输出文件
- 支持从JSON文件加载数据

## 使用方法

### 基本用法

直接运行脚本，将使用内置的示例数据：

```bash
python ttft_acc.py
```

### 从JSON文件加载数据

提供JSON数据文件路径作为命令行参数：

```bash
python ttft_acc.py sample_data.json
```

可以指定输出文件名：

```bash
python ttft_acc.py sample_data.json my_chart.png
```

### 在代码中使用

可以导入脚本并调用`plot_with_custom_data`函数：

```python
from ttft_acc import plot_with_custom_data

# 使用自定义数据
my_data = {...}  # 按照required_format组织的数据
plot_with_custom_data(custom_data=my_data, output_file='my_chart.png')
```

## 数据格式

数据应该按照以下格式组织：

```json
{
    "模型名称1": {
        "数据集名称1": {
            "缓存方法1": [x值, y值],
            "缓存方法2": [x值, y值],
            ...
        },
        "数据集名称2": {
            ...
        }
    },
    "模型名称2": {
        ...
    }
}
```

其中：
- `x值` 代表TTFT (秒)
- `y值` 代表准确率分数 (F1分数或Rouge-L分数)

样例数据文件格式可参考`sample_data.json`。

## 自定义选项

`plot_with_custom_data`函数支持以下参数：

- `custom_data`: 自定义数据字典
- `models`: 模型名称列表
- `datasets`: 数据集名称列表 
- `metrics`: 评估指标列表
- `cache_methods`: 缓存方法列表
- `output_file`: 输出文件名

## 示例

请参考`sample_data.json`中的示例数据格式。 