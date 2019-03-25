'''
You have groups that cannot be in the same bucket
Each element of a group has a label
Split the group by a percentage so that all labels are balanced in the most
maximum way
'''

labels = [
    [1, 1, 1],
    [0, 0],
    [1, 0, 1],
    [2, 1, 1],
    [2, 2, 2],
    [1],
    [1],
    [2, 1, 0]
]

features = [
    [1, 1, 1],
    [2, 2],
    [3, 3, 3],
    [4, 4, 4],
    [5, 5, 5],
    [6],
    [7],
    [8, 8, 8]
]

def get_unique_values(arr):
    values = []
    for subject in arr:
        for label in subject:
            if label not in values:
                values.append(label)
    return values

def get_value_counts(arr, values):
    count = [0] * len(values)
    for subject in labels:
        for label in subject:
            index = values.index(label)
            count[index] += 1
    return count

import random
combined = list(zip(features, labels))
random.shuffle(combined)
features[:], labels[:] = zip(*combined)

train_split = 0.7
split_index = int(len(features) * train_split)
print(len(features))
print(split_index)

train_x, train_y = features[:split_index], labels[:split_index]

test_x, test_y = features[split_index:], labels[split_index:]
print(len(train_x), len(train_y))
print(len(test_x), len(test_y))
print(train_x)
print(test_x)
print(train_y)
print(test_y)
