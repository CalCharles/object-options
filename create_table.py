import matplotlib.pyplot as plt
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

def create_bar(env, names, string_values):
    # creating the dataset
    data = {sv.split(",")[0]: sv.split(",")[1] for sv in string_values}
    costs = list(data.keys())
    num_subsets = list(data.values())
    
    fig = plt.figure(figsize = (10, 5))
    
    # creating the bar plot
    plt.bar(costs, num_subsets, color ='maroon',
            width = 0.4)
    
    plt.xlabel("Binary-subset Cost")
    plt.ylabel("No. of valid binary-subsets")
    plt.title("Cost-counts for Valid partitions for " + env)
    plt.show()


def create_histogram(env, names, string_values):
    # An "interface" to matplotlib.axes.Axes.hist() method
    n, bins, patches = plt.hist(x=d, bins='auto', color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('My Very Own Histogram')
    plt.text(23, 45, r'$\mu=15, b=3$')
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)


if __name__ == "__main__":
    string_values = list()
    with open(sys.argv[1], 'r') as f:
        for line in f.readlines():
            string_values.append(line)
    names = sys.argv[2:]
    env_name = sys.argv[1].replace("_", " ")
    create_bar(env_name, names, string_values)
    # for line in string_values:
    #     generate_latex_from_string(env_name, names, line, tabular=True)