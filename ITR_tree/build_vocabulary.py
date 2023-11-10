# 构建词汇表
class Vocabulary:
    def __init__(self):
        self.word_to_index = {}
        self.index_to_word = {}
        self.oov_index = 0  # 假设 0 是 [OOV] 的索引
        self.word_to_index['[OOV]'] = self.oov_index
        self.index_to_word[self.oov_index] = '[OOV]'
        self.index = 1

    def add_word(self, word):
        if word not in self.word_to_index:
            self.word_to_index[word] = self.index
            self.index_to_word[self.index] = word
            self.index += 1

    def get_index(self, word):
        # 获取单词的索引
        if word in self.word_to_index:
            return self.word_to_index[word]
        else:
            # 处理词汇表之外的单词
            return self.word_to_index['[OOV]']
def build_vocabulary(root, vocabulary):
    for token in root.data:
            vocabulary.add_word(token)
    # 递归地添加每个节点的单词到词汇表
    for child, edge_data in root.children:
        for token in child.data:
            vocabulary.add_word(token)
        build_vocabulary(child, vocabulary)