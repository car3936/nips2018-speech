import numpy as np

def string_to_label(x, char_to_label_dict):
    """
    :param x: string or list of string 
    :param char_to_label_dict: dict(char, label) 
    :return: np array or list of np array
    """
    label = []
    if type(x) == list:
        for i in range(len(x)):
            sentence = x[i]
            label_sentence = []
            for j in range(len(sentence)):
                label_sentence.append(char_to_label_dict[sentence[j]])
            label_sentence = np.asarray(label_sentence, dtype='int32')
            label.append(label_sentence)

        return label
    elif type(x) == str:
        for i in range(len(x)):
            label.append(char_to_label_dict[x[i]])
        label = np.asarray(label, dtype='int32')
        return label
    else:
        raise NotImplementedError

def label_to_string(x, label_to_char_dict):
    """
    x : list
    label_to_char_dict : dict(label, char)
    """
    batch_sentence = []

    if type(x[0]) == list:
        for i in range(len(x)):
            labels = x[i]
            sentence = ''
            for j in range(len(labels)):
                sentence += label_to_char_dict[labels[j]]
            batch_sentence.append(sentence)

    else:
        sentence = ''
        for j in range(len(x)):
            sentence += label_to_char_dict[x[j]]
        batch_sentence.append(sentence)

    return batch_sentence


def WER(x, y):
    """
    x, y : list of words for one sentence
    output : wer
    """

    d = np.zeros(((len(x) + 1), (len(y) + 1)), dtype='float32')

    for i in range(len(x) + 1):
        for j in range(len(y) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i
    for i in range(1, len(x) + 1):
        for j in range(1, len(y) + 1):
            if x[i - 1] == y[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitute = d[i - 1][j - 1] + 1
                insert = d[i][j - 1] + 1
                delete = d[i - 1][j] + 1
                d[i][j] = min(substitute, insert, delete)
    result = float((d[len(x)][len(y)]) / len(x)) * 100

    return result


def list_to_sparse_tensor(x, max_seq):
    """
    :param x: 2-d list (batch_size, len_seq) dtype = int
    :param max_seq: integer
    :return: indice, value, shape
    """

    batch_size = len(x)
    max_max_seq = max_seq
    for i in range(batch_size):
        if len(x[i]) > max_max_seq:
            max_max_seq = len(x[i])

    sparse_shape = np.asarray([batch_size, max_max_seq], dtype='int32')
    sparse_indices = []
    sparse_values = []

    for i in range(batch_size):
        for j in range(len(x[i])):
            sparse_indices.append([i, j])
            sparse_values.append(x[i][j])

    sparse_indices = np.asarray(sparse_indices, dtype='int32')
    sparse_values = np.asarray(sparse_values, dtype='int32')

    return sparse_indices, sparse_values, sparse_shape

def label_to_sparse_tensor(x,charset):
    """
    :param x:  list ( len_seq) dtype = int
    :param max_seq: integer
    :return: indice, value, shape
    """

    max_seq = len(x)
    _x = string_to_label(x, charset)

    sparse_shape = np.asarray([1, max_seq], dtype='int32')
    sparse_indices = []
    sparse_values = []

    for j in range(len(_x)):
        sparse_indices.append([0, j])
        sparse_values.append(_x[j])

    sparse_indices = np.asarray(sparse_indices, dtype='int32')
    sparse_values = np.asarray(sparse_values, dtype='int32')

    return sparse_indices, sparse_values, sparse_shape
if __name__ == '__main__':
    char_list = "ABCDEFGHIJKLMNOPQRSTUVWXYZ .'\n"
    charset = dict()
    for i in range(30):
        charset[char_list[i]] = i
    s1 = 'I HAVE A CAT.\n'
    label = string_to_label(s1, charset)

    print(label)
    print(label.shape)

    s2 = []
    for i in range(3):
        s2.append(s1)

    label = string_to_label(s2, charset)
    print(label)
