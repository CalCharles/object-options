import matplotlib.pyplot as plt
import sys, math, copy, os

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
    costs = [sv.split(",")[0] for sv in string_values]
    num_subsets = [math.log(float(sv.split(",")[1]), 10) + 1 for sv in string_values]
    
    fig = plt.figure(figsize = (10, 5))
    
    # creating the bar plot
    plt.bar(costs, num_subsets, color ='maroon',
            width = 1)
    plt.xlabel("Binary-subset Cost")
    plt.ylabel("No. of valid binary-subsets")
    plt.title("Cost-counts for Valid partitions for " + env)
    folder_path = os.path.join(*env.split("/")[:-1])
    plt.savefig(os.path.join(folder_path, "tables", env.split("/")[-1][:-4] + "_cost_counts.svg"))
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


def create_table(pth, names):
    string_values = list()
    with open(pth, 'r') as f:
        for line in f.readlines():
            string_values.append(line)
    hist_values = list()
    hist_str = copy.copy(pth)
    hist_str = hist_str[:hist_str.find(".txt")] + "_hist.txt"
    with open(hist_str, 'r') as f:
        for line in f.readlines():
            hist_values.append(line)
    env_name = pth.replace("_", " ")
    create_bar(env_name, names, hist_values)
    table_strings = list()
    for line in string_values:
        table_strings.append(generate_latex_from_string(env_name, names, line, tabular=True))
    folder_path = os.path.join(*pth.split("/")[:-1])
    with open(os.path.join(folder_path, "tables", pth.split("/")[-1][:-4] + "tables.txt"), 'w') as f:
        for table_str in table_strings:
            f.write(table_str)


if __name__ == "__main__":
    pth = sys.argv[1]
    names = sys.argv[2:]
    create_table(pth, names)


    # python create_table.py logs/exhaustive/ForestFire_0.01_0.2.txt April May June Fire
    # python create_table.py logs/exhaustive/ForestFire_0.01_0.2.txt April May June Fire