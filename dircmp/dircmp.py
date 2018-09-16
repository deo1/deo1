import filecmp
from pprint import pprint
import json
from os.path import normpath

class DirDifferences():
    def __init__(self, dir1: str, dir2: str) -> None:
        self.dir1 = dir1
        self.dir2 = dir2
        self.dirs_cmp = filecmp.dircmp(dir1, dir2)
        self.global_diffs = []

    def get_subdir_diffs(self, sd):
        '''Report on self and subdirs in recursively'''
        if sd.left_only or sd.right_only:
            cur_diff = {sd.left.replace('\\','/'): sd.left_only, sd.right.replace('\\','/'): sd.right_only}
            pprint(cur_diff)
            self.global_diffs.append(cur_diff)
        if sd.subdirs.values():
            for sd in sd.subdirs.values():
                self.get_subdir_diffs(sd)

if __name__ == '__main__':
    output = normpath('c:/users/jesse/desktop/diffs.json')
    onedrive = normpath('C:/Users/jesse/OneDrive/Dropbox_2018_09_13')
    dropbox = normpath('D:/Dropbox')
    dir_diffs = DirDifferences(onedrive, dropbox)
    dir_diffs.get_subdir_diffs(dir_diffs.dirs_cmp)
    diffs_json = open(output, 'w')
    json.dump(dir_diffs.global_diffs, diffs_json)
    diffs_json.close()