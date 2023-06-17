import pandas as pd


all_ = pd.read_csv("prepare_data/360x/csv/all_files.csv")
test_ = pd.read_csv("prepare_data/360x/csv/test_files_std.csv")

print("original:", all_.shape)


for i in range(18):
    print("test:", test_.iloc[i]['path'])
    s = f'path==\'{str(test_.iloc[i]["path"])}\''

    index = all_.query(s).index[0]
    print("all:", all_.iloc[index]['path'], "\n")

    all_ = all_.drop(index=index, axis=1)
    all_.reset_index(inplace=True, drop=True)


test_.to_csv("prepare_data/360x/csv/test_files_test.csv", index=False)
all_.to_csv("prepare_data/360x/csv/train_files_tes.csv", index=False)

print("process:", all_.shape)

