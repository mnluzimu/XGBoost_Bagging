

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
        if entry_list[0] == 'e':
            label.append(1)
        else:
            label.append(0)

        data_entry = []
        if entry_list[1] == 'b':
            data_entry.append(0)
        elif entry_list[1] == 'c':
            data_entry.append(1)
        elif entry_list[1] == 'x':
            data_entry.append(2)
        elif entry_list[1] == 'f':
            data_entry.append(3)
        elif entry_list[1] == 'k':
            data_entry.append(4)
        elif entry_list[1] == 's':
            data_entry.append(5)

        if entry_list[2] == 'f':
            data_entry.append(0)
        elif entry_list[2] == 'g':
            data_entry.append(1)
        elif entry_list[2] == 'y':
            data_entry.append(2)
        elif entry_list[2] == 's':
            data_entry.append(3)
        elif entry_list[2] == 'n':
            data_entry.append(4)
        elif entry_list[2] == 'b':
            data_entry.append(5)
        elif entry_list[2] == 'c':
            data_entry.append(6)
        elif entry_list[2] == 'g':
            data_entry.append(7)
        elif entry_list[2] == 'r':
            data_entry.append(8)


    print(data, label)

    return data, label


if __name__ == '__main__':
    loadData('agaricus-lepiota.data')
