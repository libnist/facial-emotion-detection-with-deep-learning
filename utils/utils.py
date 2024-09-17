from pathlib import Path

def get_paths(path):
    path = Path(path)
    data_paths = []
    targets = {}
    i = 0
    for root, dirs, files in path.walk():
        if len(files) > 0:
            paths = [(root/file, i) for file in files]
            data_paths += paths
            targets[root.name] = i
            i += 1
            
    return data_paths, targets    