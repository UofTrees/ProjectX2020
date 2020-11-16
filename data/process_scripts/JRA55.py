import os
import pandas as pd
import tqdm
import glob
import argparse

parser = argparse.ArgumentParser(description="JRA55")
parser.add_argument("--trial", type=int)

args = parser.parse_args()

root_dir = "trial" + str(args.trial)
os.chdir(root_dir)

def unzip_del(dir_name):
    os.chdir("./" + dir_name)
    all_files = glob.glob("./*.tar")
    for filename in all_files:
        os.system("tar -xf {}".format(filename))
#     os.system("rm -f {}/*.tar".format(dir_name))
#     os.system("rm {}/*.py".format(dir_name))
    os.chdir("../")
    
def concat_files(dir_name, colname):
    all_files = glob.glob(dir_name + "/*.csv")
    data_list = []
    for filename in all_files:
        df = pd.read_csv(filename, header=0, index_col=False)
        df = df.iloc[1:]
        data_list.append(df)
    data = pd.concat(data_list, axis=0)
    data.rename(columns={'ParameterValue': colname}, inplace=True)
    data = data.sort_values(by='ValidDate')
    print(data.shape)
    print(list(data.columns))
    return data

if __name__ == "__main__":
    unzip_del("subset1")
    unzip_del("subset2")
    unzip_del("subset3")

    data1 = concat_files("subset1", "RH")
    data2 = concat_files("subset2", "CM")
    data3 = concat_files("subset3", "CT")

    data = pd.merge(data1, data2, how='inner', on=['Longitude', 'Latitude', 'ValidDate', 'ValidTime'])
    data = pd.merge(data, data3, how='inner', on=['Longitude', 'Latitude', 'ValidDate', 'ValidTime'])

    print(data.columns)

    data = data.drop(columns=['#ParameterName', '#ParameterName_x', '#ParameterName_y',
                   'LevelValue', 'LevelValue_x', 'LevelValue_y'])

    print(data.head())

    data.to_csv("raw.csv", index=False)
