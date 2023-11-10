class TreeNode:
    def __init__(self, data, tag):
        self.data = data
        self.tag= tag
        self.children = []
        self.embedding = []
        self.deep = 0


# 给树添加边的信息
def add_edge(parent, child, edge_data):
    parent.children.append((child, edge_data))
    
def build_ITR_tree(Seqsstate, Seqslog, Seqscall_1):
    # 创建节点字典，将调用ID映射到相应的树节点
    call_id_to_node = {}
    
    # 初始化根节点
    root = TreeNode(Seqscall_1[0][0],'call')
    call_id_to_node[Seqscall_1[0][0]] = root
    root.data = Seqscall_1[1][0]
    root.deep = 0
    # 遍历调用序列以构建树
    for i in range(1, len(Seqscall_1[0])):
        call = Seqscall_1[0][i]
        parent_call = Seqscall_1[0][i - 1]
        parent_node = call_id_to_node[parent_call]
        call_node = TreeNode(call,'call')
        call_node.data = Seqscall_1[1][i]
        call_node.deep = parent_node.deep +1
        # 添加调用之间的边信息
        add_edge(parent_node, call_node, f"Call {parent_call} -> {call}")
        
        call_id_to_node[call] = call_node
    
    # 遍历状态跟踪以构建树
    i = 0
    for state_call in Seqsstate[0]:
        state_node = TreeNode(state_call,'state')
        parent_call = Seqsstate[1][state_call]
        parent_node = call_id_to_node[parent_call]
        
        # 添加状态与调用之间的边信息
        add_edge(parent_node, state_node, f"State {parent_call} -> {state_call}")
        state_node.data = Seqsstate[2][i]
        state_node.deep = parent_node.deep + 1
        i = i+1

    # 遍历日志跟踪以构建树
    i = 0
    for log_call in Seqslog[0]:
        log_node = TreeNode(log_call,'log')
        parent_call = Seqslog[1][log_call]
        parent_node = call_id_to_node[parent_call]
        
        # 添加日志与调用之间的边信息
        add_edge(parent_node, log_node, f"Log {parent_call} -> {log_call}")
        log_node.data = Seqslog[2][i]
        log_node.deep = parent_node.deep + 1
        i = i+1

    return root