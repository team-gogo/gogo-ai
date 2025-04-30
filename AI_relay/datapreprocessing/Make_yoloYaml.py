# yolo.yaml 파일 생성
import yaml

data = {
    "train" : 'G:\\다른 컴퓨터\\내 노트북\\ai\\gogo v3\\AI_relay\\data\\datasets\\Training\\',
        "val" : 'G:\\다른 컴퓨터\\내 노트북\\ai\\gogo v3\\AI_relay\\data\\datasets\\Validation\\',
        "names" : {1 : 'ball'}}

with open('AI_relay\\yolo.yaml', 'w') as f :
    yaml.dump(data, f)

with open('AI_relay\\yolo.yaml', 'r') as f :
    lines = yaml.safe_load(f)
    print(lines)