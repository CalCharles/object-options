import sys

def generate_latex_from_string(env, names, string_values, tabular = False):
    if tabular:
        table_string = """\\begin{tabular}"""
    else:
        table_string = """\\begin{table}[h]
\\setlength\\tabcolsep{2pt}
\\vskip 0.2cm
\\begin{center}
\\begin{small}
\\begin{tabular}"""
    table_string += "{l" + "|c" * (len(names) - 1) + "|r}" + "\n\\toprule\n"
    table_string += "& ".join(["Binary"] + names) + "\\\\\n\midrule\n"
    for str_row in string_values.split(";"):
        table_string += "$" + "$& $".join(str_row.split(",")) + "$\\\\\n"
    table_string = table_string[:-5] + "$\\\\\n"
    table_string += "\\bottomrule\n\\end{tabular}\n"
    if not tabular:
        table_string += "\caption{One of the min cost subset-binary partition for " + env + ". The combination should subsume all actual causes}\n"
        table_string += '''\label{Evaluation Table}
\end{small}
\end{center}
\end{table}
'''
    print(table_string)
    return table_string

if __name__ == "__main__":
    string_values = list()
    with open(sys.argv[1], 'r') as f:
        for line in f.readlines():
            string_values.append(line)
    names = sys.argv[2:]
    env_name = sys.argv[1].replace("_", "\\_")
    for line in string_values:
        generate_latex_from_string(env_name, names, line, tabular=True)