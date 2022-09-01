

def loadData(fileName):
    """
    :param fileName: name of input file
    :return: data, value
    """
    fp = open(fileName, encoding='utf-8')
    # 存放数据向量
    data = []
    # 存放标签值
    label = []
    for line in fp:
        entry_list = line.strip().split(',')
    print(data, label)

    return data, label


if __name__ == '__main__':
    loadData('agaricus-lepiota.data')
