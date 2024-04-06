# -*- coding: utf-8 -*-
import requests
from bs4 import BeautifulSoup

def read_var_from_file():
    with open('../parameter/var.txt', 'r') as f:
        return f.read()

def read_total_voxels_from_file():
    with open('../parameter/total_voxels.txt', 'r') as f:
        return f.read()


def cal_r():
    voxel = 1
    total_voxels = read_total_voxels_from_file()
    total_voxels = int(total_voxels)
    voxel_volume = voxel * voxel * voxel
    # 实际体积 以立方厘米为单位
    total_cubic_cm = total_voxels * voxel_volume / 1000
    value = read_var_from_file()
    value = int(value)  # 此时就是实际像素点的体积
    Proportion = value / total_voxels
    # 计算大小
    tumor_volume = total_cubic_cm * Proportion  # 此刻的单位是立方厘米
    r = (tumor_volume * 3 / 4 / 3.14) ** (1 / 3)
    diameter = 2 * r + 0.1
    diameter = round(diameter, 4)  # 单位是cm  最大径
    return diameter

def url():
    search_terms = "肺癌 最大径为"
    # 设置请求头，模仿浏览器行为
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }

    diameter = cal_r()
    search_terms = search_terms + str(diameter)
    search_url = 'https://www.baidu.com/s'

    params = {'wd': search_terms}

    str = ""
    # 发送请求
    response = requests.get(search_url, headers=headers, params=params)

    # 确保请求成功
    if response.status_code == 200:
        # 解析HTML内容
        soup = BeautifulSoup(response.text, 'html.parser')

        search_results = soup.find_all('h3', class_='t')

        # 打印每个搜索结果的URL
        for result in search_results:
            link = result.find('a')
            if link and link['href']:
                str = str + link
                str = str + '\n'
                print(link['href'])
        return str


    else:
        print("请求失败，状态码：", response.status_code)