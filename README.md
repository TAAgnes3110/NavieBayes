# Naive Bayes Project

## Mô tả

Dự án này xây dựng mô hình phân loại Naive Bayes để dự đoán kết quả học tập của sinh viên (Graduate, Dropout, Enrolled) dựa trên các thông tin cá nhân, học tập, tài chính, gia đình, v.v.

## Cấu trúc thư mục

```
naive-bayes-project/
│
├── data/
│   └── data.csv              # Dữ liệu đầu vào (không có header, cột cuối là nhãn)
│
├── output/
│   ├── naive_bayes_model.pkl # File lưu mô hình đã huấn luyện
│   └── predictions.csv       # Kết quả dự đoán trên tập test
│
├── src/
│   ├── NavieBayes.py         # Cài đặt thuật toán Naive Bayes
│   ├── train.py              # Huấn luyện và lưu mô hình
│   └── evaluate.py           # Đánh giá mô hình, xuất file dự đoán
│
└── README.md
```

## Hướng dẫn sử dụng

### 1. Chuẩn bị dữ liệu

- Đặt file `data.csv` vào thư mục `data/`.
- Mỗi dòng là một mẫu, cột cuối là nhãn (`Graduate`, `Dropout`, `Enrolled`).

### 2. Huấn luyện mô hình

Chạy lệnh sau trong thư mục `src`:

```bash
python train.py
```

- Mô hình sẽ được lưu tại `output/naive_bayes_model.pkl`.

### 3. Đánh giá và xuất kết quả dự đoán

Chạy lệnh:

```bash
python evaluate.py
```

- Kết quả dự đoán sẽ được lưu tại `output/predictions.csv`.

### 4. Cài đặt phụ thuộc

- Python >= 3.7
- pandas

Cài đặt nhanh:

```bash
pip install pandas
```
