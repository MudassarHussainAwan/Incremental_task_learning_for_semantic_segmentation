
tasks_voc = {
    
    "offline-ov":
        {
            0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        },
    "15-5-ov":
        {
            0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            1: [0, 16, 17, 18, 19, 20]
        },
    "15-5s-ov":
        {
            0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            1: [0, 16],
            2: [0, 17],
            3: [0, 18],
            4: [0, 19],
            5: [0, 20]
        }
}


def get_task_labels(dataset, name, step):
    if dataset == 'voc':
        task_dict = tasks_voc[name]
    else:
        raise NotImplementedError
    assert step in task_dict.keys(), f"You should provide a valid step! [{step} is out of range]"

    labels = list(task_dict[step])
    labels_c = []
    for i in range(step+1):
        labels_c.append(task_dict[i])
    labels_cum = list(set(x for lst in labels_c for x in lst))
    return labels,labels_cum,f'/content/root_drive/MyDrive/Project/splits/{name}'

def get_steps(name):
    return list(tasks_voc[name].keys())
def get_classes(name):
    x = [16,6]
    y = [16,2,2,2,2]
    if name == '15-5-ov':
      return x
    elif name == '15-5s-ov':
      return y
    else:
      raise NotImplementedError

if __name__ == '__main__':
    print(get_task_labels('voc','15-5s-ov', 3))


