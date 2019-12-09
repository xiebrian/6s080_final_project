import pandas as pd
import bz2
import json
from tqdm import tqdm
import argparse

def run(day):
    columns = ['created_at', 'id', 'text', 'lang', 'in_reply_to_screen_name', 'retweeted_status']
    # columns_df = pd.DataFrame(columns)
    # columns_df.to_csv("data/01.csv", index=False)

    for i in range(24):
        data = []
        str_i = "0" + str(i) if i < 10 else str(i)
        for j in range(60):
            str_j = "0" + str(j) if j < 10 else str(j)
            try:
                with bz2.open("data/" + day + "/" + str_i + "/" + str_j + ".json.bz2", "rt") as bzinput:
                    lines = bzinput.readlines()
            except FileNotFoundError:
                continue
            for line in range(len(lines)):
                json_line = json.loads(lines[line])

                relevant_columns = []
                for column in columns[:len(columns) - 1]:
                    if column in json_line:
                        relevant_columns.append(json_line[column])
                    else:
                        relevant_columns.append('')
                if 'retweeted_status' in json_line:
                    relevant_columns.append("True")
                else:
                    relevant_columns.append("False")
                data.append(relevant_columns)

        print("written " + str_i + "!")
        df = pd.DataFrame(data)
        df.to_csv("data/01" + day + ".csv", index=False, mode='a')
    print("written 2!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("i")
    args = parser.parse_args()

    run(args.i)
