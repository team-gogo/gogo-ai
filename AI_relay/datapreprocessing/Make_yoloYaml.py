# yolo.yaml 파일 생성
import yaml

data = {
    "train" : 'C:\\Users\\kdyeo\\gogo\\Training\\',
        "val" : 'C:\\Users\\kdyeo\\gogo\\Validation\\',
        "names" : {0: 'ball'}}

with open('G:\\다른 컴퓨터\\내 노트북\\ai\\gogo v3\\AI_relay\\yolo.yaml', 'w') as f :
    yaml.dump(data, f)

with open('G:\\다른 컴퓨터\\내 노트북\\ai\\gogo v3\\AI_relay\\yolo.yaml', 'r') as f :
    lines = yaml.safe_load(f)
    print(lines)