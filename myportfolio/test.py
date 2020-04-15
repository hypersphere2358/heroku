class Dir(object):
    def __init__(self, name):
        self.name = name
        self.sub = []

    def addSubDir(self, obj):
        if obj.name in self.subNames():
            return
        self.sub.append(obj)

    def subNames(self):
        names = []
        for s in self.sub:
            names.append(s.name)
        return names


def solution(directory, command):
    answer = []

    dir_tree = Dir("/")

    new_dir = Dir("root")

    for i in range(len(directory)):
        dir_list = directory[i].split("/")
        dir_list = dir_list[1:]

        if dir_list[0] != "":
            print(dir_list)
            cur_dir = dir_tree
            for j in range(len(dir_list)):
                new_dir = Dir(dir_list[j])
                cur_dir.addSubDir(new_dir)
                cur_dir = new_dir

    print(dir_tree.subNames())

    return answer