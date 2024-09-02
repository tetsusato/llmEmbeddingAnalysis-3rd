import re
from collections import defaultdict
import polars as pl

def parse_log_line(line):
    # A) と B) のメッセージを区別するための正規表現
    pattern_a = re.compile(
        r'^.+ -> Results, mode, model, num_inputs, start_index,stride, time_delay, reduce_func, n\n'
        r'.+ -> ([\w_\d]+), ([\d\.\-]+), (\w+), (\w+), (\d+),(\d+), (\d+), (\d+), ([\w_]+), (\d+)'
        #r'^(.+) -> (.*)'
        #r'(.+) -> (.*)'
    )
    pattern_b = re.compile(
        r'^.+ -> Results, mode, model, num_inputs, start_index,perplexity, reduce_func\n'
        r'.+ -> ([\w_\d]+), ([\d\.\-]+), (\w+), (\w+), (\d+),(\d+), (\d+), ([\w_]+)'
    )
    #print(pattern_a)
    #print(line)
    match_a = pattern_a.match(line)
    #print(match_a)
    #if match_a:
    #    print(match_a.groups())
    match_b = pattern_b.match(line)

    if match_a:
        return {
            'type': 'A',
            'corr_index': match_a.group(1),
            'results_corr': float(match_a.group(2)),
            'mode': match_a.group(3),
            'model': match_a.group(4),
            'num_inputs': int(match_a.group(5)),
            'start_index': int(match_a.group(6)),
            'stride': int(match_a.group(7)),
            'time_delay': int(match_a.group(8)),
            'reduce_func': match_a.group(9),
            'n': int(match_a.group(10)),
        }
    elif match_b:
        return {
            'type': 'B',
            'corr_index': match_b.group(1),
            'results_corr': float(match_b.group(2)),
            'mode': match_b.group(3),
            'model': match_b.group(4),
            'num_inputs': int(match_b.group(5)),
            'start_index': int(match_b.group(6)),
            'perplexity': int(match_b.group(7)),
            'reduce_func': match_b.group(8),
        }
    return None

def parse_log_file(log_file_path):
    resultsA = []
    resultsB = []
    with open(log_file_path, 'r') as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            if 'Results' in lines[i]:
                entry = parse_log_line(''.join(lines[i:i+2]))
                if entry:
                    if entry["type"] == "A":
                        resultsA.append(entry)
                    elif entry["type"] == "B":
                        resultsB.append(entry)
                i += 2
            else:
                i += 1
    return resultsA, resultsB

# 使用例
log_file_path = './results_fit2024.log'
parsed_resultsA, parsed_resultsB = parse_log_file(log_file_path)
print(parsed_resultsA)
print(parsed_resultsB)

# 解析結果を表示
#for result in parsed_results:
#    print(result)
    

resultsA_pl = pl.DataFrame(parsed_resultsA)
resultsB_pl = pl.DataFrame(parsed_resultsB)

print(resultsA_pl)
print(resultsB_pl)


aveA = resultsA_pl.group_by("corr_index").agg([
        pl.col("results_corr").mean().alias("average_corr")
    ])
pl.Config().set_tbl_cols(1)
pl.Config(tbl_cols=10)
print(aveA)
aveB = resultsB_pl.group_by("corr_index").agg([
        pl.col("results_corr").mean().alias("average_corr")
    ])

print(aveB)
