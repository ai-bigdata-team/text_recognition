import os
import gdown
import yaml
import numpy as np
import uuid
import requests
import tempfile
from tqdm import tqdm

def download_weights(uri, cached=None, md5=None, quiet=False):
    if uri.startswith('http'):
        return download(url=uri, quiet=quiet)
    return uri

def download(url, quiet=False):
    tmp_dir = tempfile.gettempdir()
    filename = url.split('/')[-1]
    full_path = os.path.join(tmp_dir, filename)
    
    if os.path.exists(full_path):
        print('Model weight {} exsits. Ignore download!'.format(full_path))
        return full_path

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(full_path, 'wb') as f:
            for chunk in tqdm(r.iter_content(chunk_size=8192)):
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None.
                #if chunk:
                f.write(chunk)
    return full_path

def download_config(id):
    url = 'https://vocr.vn/data/vietocr/config/{}'.format(id)
    r = requests.get(url)
    config = yaml.safe_load(r.text)
    return config

def levenshtein_distance(a, b):
    """Simple Levenshtein distance (fallback nếu không cài python-Levenshtein)."""
    dp = np.zeros((len(a) + 1, len(b) + 1), dtype=np.int32)

    for i in range(len(a) + 1):
        dp[i][0] = i
    for j in range(len(b) + 1):
        dp[0][j] = j

    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # deletion
                dp[i][j-1] + 1,      # insertion
                dp[i-1][j-1] + cost  # substitution
            )
    return dp[-1][-1]

def compute_CER(ground_truth, predictions):
    """
    CER = (S + D + I) / N
    S, D, I được tính bằng Levenshtein distance.
    """
    total_dist = 0
    total_chars = 0

    for gt, pred in zip(ground_truth, predictions):
        dist = levenshtein_distance(gt, pred)
        total_dist += dist
        total_chars += len(gt)

    if total_chars == 0:
        return 0.0

    return total_dist / total_chars


def compute_WER(ground_truth, predictions):
    """
    WER = (S + D + I) / W
    Tương tự CER nhưng tính theo từ.
    """
    total_dist = 0
    total_words = 0

    for gt, pred in zip(ground_truth, predictions):
        gt_words = gt.split()
        pred_words = pred.split()

        dist = levenshtein_distance(gt_words, pred_words)
        total_dist += dist
        total_words += len(gt_words)

    if total_words == 0:
        return 0.0

    return total_dist / total_words

def compute_accuracy(ground_truth, predictions, mode='full_sequence'):
    """
    Computes accuracy
    :param ground_truth:
    :param predictions:
    :param display: Whether to print values to stdout
    :param mode: if 'per_char' is selected then
                 single_label_accuracy = correct_predicted_char_nums_of_single_sample / single_label_char_nums
                 avg_label_accuracy = sum(single_label_accuracy) / label_nums
                 if 'full_sequence' is selected then
                 single_label_accuracy = 1 if the prediction result is exactly the same as label else 0
                 avg_label_accuracy = sum(single_label_accuracy) / label_nums
    :return: avg_label_accuracy
    """
    if mode == 'per_char':

        accuracy = []

        for index, label in enumerate(ground_truth):
            prediction = predictions[index]
            total_count = len(label)
            correct_count = 0
            try:
                for i, tmp in enumerate(label):
                    if tmp == prediction[i]:
                        correct_count += 1
            except IndexError:
                continue
            finally:
                try:
                    accuracy.append(correct_count / total_count)
                except ZeroDivisionError:
                    if len(prediction) == 0:
                        accuracy.append(1)
                    else:
                        accuracy.append(0)
        avg_accuracy = np.mean(np.array(accuracy).astype(np.float32), axis=0)
    elif mode == 'full_sequence':
        try:
            correct_count = 0
            for index, label in enumerate(ground_truth):
                prediction = predictions[index]
                if prediction == label:
                    correct_count += 1
            avg_accuracy = correct_count / len(ground_truth)
        except ZeroDivisionError:
            if not predictions:
                avg_accuracy = 1
            else:
                avg_accuracy = 0
    elif mode =='cer':
        avg_accuracy = compute_CER(ground_truth, predictions)
    elif mode == 'wer':
        avg_accuracy = compute_WER(ground_truth, predictions)
    else:
        raise NotImplementedError('Other accuracy compute mode has not been implemented')
    return avg_accuracy
