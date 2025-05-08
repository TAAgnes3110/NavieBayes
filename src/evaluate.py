import os
import pickle
import pandas as pd
import csv
from collections import Counter
from NavieBayes import predict

def load_data(filepath):
    df = pd.read_csv(filepath)
    df = df.fillna(df.mode().iloc[0])
    features = list(df.columns[:-1])
    samples = df.values.tolist()
    return samples, features, df

def accuracy(y_true, y_pred):
    correct = sum(yt == yp for yt, yp in zip(y_true, y_pred))
    return correct / len(y_true) if y_true else 0

if __name__ == "__main__":
    data_path = os.path.join('..', 'data', 'data.csv')
    if not os.path.exists(data_path):
        print(f"File {data_path} không tồn tại.")
        exit(1)

    data, features, df = load_data(data_path)

    # Chia train/test (80% train, 20% test)
    split_idx = int(0.8 * len(data))
    train_data = data[:split_idx]
    test_data = data[split_idx:]

    # Load mô hình đã lưu
    model_path = os.path.join('..', 'output', 'naive_bayes_model.pkl')
    with open(model_path, 'rb') as f:
        prior_prob, cond_prob, features_model = pickle.load(f)

    y_true = [row[-1] for row in test_data]
    y_pred = [predict(prior_prob, cond_prob, row[:-1]) for row in test_data]

    acc = accuracy(y_true, y_pred)
    print(f"Accuracy trên tập test: {acc:.2%}")

    print("\nPhân bố dự đoán trên tập test:")
    for label, count in Counter(y_pred).items():
        print(f"{label}: {count} mẫu")

    print("\nPhân bố thực tế trên tập test:")
    for label, count in Counter(y_true).items():
        print(f"{label}: {count} mẫu")

    # Tạo thư mục output nếu chưa có
    output_dir = os.path.join('..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'predictions.csv')

    # Ghi file dự đoán
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(features + ['TrueLabel', 'PredictedLabel'])
        for row, true, pred in zip(test_data, y_true, y_pred):
            writer.writerow(row[:-1] + [true, pred])

    print(f"\nĐã lưu kết quả dự đoán vào: {output_path}")
