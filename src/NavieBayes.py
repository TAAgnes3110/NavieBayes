import math
from collections import Counter

def split_data(data, feature_index, value):
    "Chia dữ liệu thành một tập con dựa trên giá trị của một thuộc tính"
    return [row for row in data if row[feature_index] == value]

def probability(data, feature_index, value):
    "Tính xác suất của một thuộc tính trong dữ liệu"
    total_count = len(data)
    value_count = sum(1 for row in data if row[feature_index] == value)
    return value_count / total_count if total_count > 0 else 0.0

def likelihood(data, feature_index, value, label, label_counts, smoothing=1):
    "Tính xác suất có điều kiện P(X|Y) với Laplace smoothing"
    subset = [row for row in data if row[-1] == label]
    value_count = sum(1 for row in subset if row[feature_index] == value)
    feature_values = set(row[feature_index] for row in data)
    # Laplace smoothing
    return (value_count + smoothing) / (len(subset) + smoothing * len(feature_values))

def train_naive_bayes(data):
    "Huấn luyện mô hình Naive Bayes (Discrete, Laplace smoothing)"
    labels = [row[-1] for row in data]
    label_counts = Counter(labels)
    total_count = len(data)

    # Tính xác suất cho mỗi nhãn
    prior_probabilities = {label: count / total_count for label, count in label_counts.items()}

    # Tính xác suất có điều kiện cho mỗi thuộc tính
    conditional_probabilities = {}
    for feature_index in range(len(data[0]) - 1):  # Không tính nhãn
        conditional_probabilities[feature_index] = {}
        feature_values = set(row[feature_index] for row in data)
        for label in label_counts:
            conditional_probabilities[feature_index][label] = {}
            for value in feature_values:
                conditional_probabilities[feature_index][label][value] = likelihood(
                    data, feature_index, value, label, label_counts
                )

    return prior_probabilities, conditional_probabilities

def predict(prior_probabilities, conditional_probabilities, input_data):
    "Dự đoán nhãn cho dữ liệu đầu vào (tính log xác suất để tránh underflow)"
    label_scores = {}
    for label, prior in prior_probabilities.items():
        score = math.log(prior) if prior > 0 else float('-inf')
        for feature_index, value in enumerate(input_data):
            prob = conditional_probabilities[feature_index][label].get(value, 1e-9)
            score += math.log(prob)
        label_scores[label] = score
    return str(max(list(label_scores), key=lambda k: label_scores[k]))
