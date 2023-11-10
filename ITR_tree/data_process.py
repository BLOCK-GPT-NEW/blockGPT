
#数据处理部分函数
def process_call_entry(call_entry, call_idx, Seqscall_1, Seqsstate_1, Seqslog_1, state_idx, log_idx):
    Seqscall_1[0].append(f't{call_idx}call')
    call_info = [f'{call_entry["call_from"]},{call_entry["call_to"]},{call_entry["call_function_name"]},{call_entry["call_gas"]},{call_entry["call_value"]}']

    for input_entry in call_entry["call_input"]:
        call_info.extend([f'{input_entry["call_input_type"]},{input_entry["call_input_value"]}'])

    for output_entry in call_entry["call_output"]:
        call_info.extend([f'{output_entry["call_output_type"]},{output_entry["call_output_value"]}'])

    Seqscall_1[1].extend([','.join(call_info)])

    for state_entry in call_entry["stata"]:
        Seqsstate_1[0].append(f't{state_idx}state')
        Seqsstate_1[1][f't{state_idx}state'] = f't{call_idx}call'
        state_info = [f'{state_entry["tag"]},{state_entry["key"]},{state_entry["value"]}']
        Seqsstate_1[2].extend(state_info)
        state_idx += 1

    for log_entry in call_entry["log"]:
        Seqslog_1[0].append(f't{log_idx}log')
        Seqslog_1[1][f't{log_idx}log'] = f't{call_idx}call'
        log_info = [f'{log_entry["contract_address"]},{log_entry["event_hash"]}']

        for l_d_entry in log_entry["data"]:
            log_info.extend([f'{l_d_entry["type"]},{l_d_entry["value"]}'])

        Seqslog_1[2].extend(log_info)
        log_idx += 1
    return state_idx,log_idx
def process_entry(entry):
    Seqscall_1 = [[],[]]
    Seqsstate_1 = [[],{},[]]
    Seqslog_1 = [[],{},[]]

    state_idx = 0
    log_idx = 0

    for call_idx, call_entry in enumerate(entry["call"]):
        state_idx,log_idx = process_call_entry(call_entry, call_idx, Seqscall_1, Seqsstate_1, Seqslog_1, state_idx, log_idx)

    return Seqsstate_1, Seqslog_1, Seqscall_1