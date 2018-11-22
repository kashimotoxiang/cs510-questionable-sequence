import lyx
import collections


def create_freq():
    en_freq = {}
    lines = lyx.io.read_all_lines("en_freq.txt")
    pre_count = 0
    rank = 0
    for line in lines:
        tmp = line.split(" ")
        word = tmp[0]
        count = tmp[1]
        if pre_count != count:
            pre_count = count
            rank += 1
        en_freq[word] = rank
    lyx.io.save_pkl(en_freq, "en_freq")


if __name__ == "__main__":
    create_freq()
