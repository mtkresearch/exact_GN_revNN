import csv
from distutils.dir_util import copy_tree
import os


optimizer = "results_ggn_mlp25M"
base_dir = f"/proj/gpu_mtk53548/Fastbreak/research/{optimizer}"

best_loss_after_1_epoch = 10000
best_conf_name = ""
for res_folder in os.listdir(base_dir):
    print(res_folder)
    cur_conf_name = res_folder
    cur_path = os.path.join(base_dir, cur_conf_name)
    for inside_folder in os.listdir(cur_path):
        res_csv_file = os.path.join(
            cur_path, inside_folder, "per_epoch", "progress.csv"
        )
        with open(res_csv_file) as csvfile:
            spamreader = csv.reader(csvfile, delimiter=",")
            for idx, row in enumerate(spamreader):
                if idx == 1:
                    cur_loss_after_1_epoch = row[4]
                    if cur_loss_after_1_epoch == "nan":
                        cur_loss_after_1_epoch = 1000000
                    else:
                        cur_loss_after_1_epoch = eval(row[4])
                    break

    if cur_loss_after_1_epoch < best_loss_after_1_epoch:
        best_loss_after_1_epoch = cur_loss_after_1_epoch
        best_conf_name = cur_conf_name

best_res_path = os.path.join(base_dir, best_conf_name)
print("Best configuration is:", best_res_path)
copy_tree(best_res_path, f"best_{optimizer}")
