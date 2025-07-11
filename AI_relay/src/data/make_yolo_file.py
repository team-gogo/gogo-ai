# yolo.yaml 파일 생성
import yaml

data = {
    "train" : '../../data/datasets/Training',
    "val" : '../../data/datasets/Validation',
    "test" : '../../data/datasets/Test',
    "nc" : 3,
        "names" : {0: 'ball', 1: 'player', 2: 'goal'},}

with open('../../config/yolo.yaml', 'w') as f :
    yaml.dump(data, f)

with open('../../config/yolo.yaml', 'r') as f :
    lines = yaml.safe_load(f)
    print(lines)