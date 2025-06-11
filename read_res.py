"""
此脚本用于读取指定路径下的 progress.txt 文件，并将其内容进行解析。
解析后的数据存储在字典列表中，方便后续处理和分析。
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_progress_file(file_path: str) -> list[dict]:
    """
    读取 progress.txt 文件并解析其内容。

    Args:
        file_path (str): 文件的完整路径。

    Returns:
        list[dict]: 包含每行数据解析结果的字典列表。
    """
    if not os.path.exists(file_path):
        print(f"文件 {file_path} 不存在。")
        return []

    data = []
    headers = None
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                if not headers:
                    # 第一行作为表头
                    headers = line.split()
                else:
                    # 后续行作为数据
                    values = line.split()
                    row_data = dict(zip(headers, values))
                    data.append(row_data)
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")

    return data

def draw_result_ret(file_path):

    # 定义文件路径
    read_file_path = file_path + "/progress.txt"

    # 调用函数读取文件
    progress_data = read_progress_file(read_file_path)

    progress_data = pd.DataFrame(progress_data)

    # EpRet = progress_data['StdEpRet'].values.astype(float)
    # TestEpRet = progress_data['StdTestEpRet'].values.astype(float)
    EpRet = progress_data['AverageEpRet'].values.astype(float)
    TestEpRet = progress_data['AverageTestEpRet'].values.astype(float)

    plt.figure()
    plt.plot(EpRet, label='EpRet')
    plt.plot(TestEpRet, label='TestEpRet')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(file_path + '/EpRet.png')


def draw_result_net(file_path):

    # 定义文件路径
    read_file_path = file_path + "/progress.txt"

    # 调用函数读取文件
    progress_data = read_progress_file(read_file_path)

    progress_data = pd.DataFrame(progress_data)

    QVals = progress_data['AverageQVals'].values.astype(float)
    LossPi = progress_data['LossPi'].values.astype(float)
    LossQ = progress_data['LossQ'].values.astype(float)

    plt.figure()
    plt.plot(QVals, label='QVals')
    plt.xlabel('Epoch')
    plt.savefig(file_path + '/QVals.png')

    plt.figure()
    plt.plot(LossPi, label='LossPi')
    plt.plot(LossQ, label='LossQ')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(file_path + '/Loss.png')

# draw_result(file_path)