# yolo.yaml 파일 생성
import yaml

data = {
    "path" : 'G:\\다른 컴퓨터\\내 노트북\\ai\\gogo v3\\AI_relay\\data\\datasets',
    "train" : '\\Training\\',
        "val" : '\\Validation\\',
        "names" : {0 : 'player', 1 : 'ball', 2 : 'goalpost'}}

with open('G:\\다른 컴퓨터\\내 노트북\\ai\\gogo v3\\AI_relay\\yolo.yaml', 'w') as f :
    yaml.dump(data, f)

with open('G:\\다른 컴퓨터\\내 노트북\\ai\\gogo v3\\AI_relay\\yolo.yaml', 'r') as f :
    lines = yaml.safe_load(f)
    print(lines)