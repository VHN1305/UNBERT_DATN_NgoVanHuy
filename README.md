# UNBERT

UNBERT là mô hình gợi ý tin tức được xây dựng trên nền tảng BERT. Mô hình được huấn luyện trên tập dữ liệu MIND-small.

## Requirements

Để chạy đồ án, vui lòng cài đặt python3.8 và các thư viện cần thiết bằng lệnh sau:

```bash
pip install -r requirements.txt
```

## Data preparation

Với tập MIND dataset, hãy tải data ở: https://msnews.github.io

| File Name        | Description              |
| ---------------- | ------------------------ |
| data/small/train | MIND-small train dataset |
| data/small/dev   | MIND-small dev dataset   |

## Usage

```python
python run.py --mode train --split small --root ./data/ --pretrain bert-base-uncased/
```

Điều chỉnh nhiều tham số hơn trong file `run.py`

## Run Web Demo

```python
python app.py
```
