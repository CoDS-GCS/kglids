import io
import json
import time

import pandas as pd
from subprocess import check_output, CalledProcessError, STDOUT
import shlex
import os
from tqdm import tqdm
import random  # change this to the real value
import kaggle  # make sure kaggle package is downloaded

# to keep first 10 rows of dataset:
# for i in *.csv; do echo "$(head -10 $i)" > $i; done


def syscall(command):
    """
    params:
        command: string, ex. `"ls -l"`
    returns: output, success
    """
    command = shlex.split(command)
    try:
        output = check_output(command, stderr=STDOUT).decode()
        success = True
    except CalledProcessError as e:
        output = e.output.decode()
        success = False
    return output, success


def main():

    datasets = pd.read_csv('dataset_csv.csv')

    for _, row in tqdm(datasets.iterrows(), total=len(datasets)):
        dataset_name = row['ref']
        dataset_path = 'data/kaggle/' + dataset_name.replace('/', '.')

        # Download the dataset (if size is < 100MB)
        size = row['size']
        if 'GB' in size or ('MB' in size and int(size[:size.index('MB')]) > 100):
            print('Dataset too big:', dataset_name, size)
            continue
        else:
            download_cmd = f'kaggle datasets download {dataset_name} --path {dataset_path}/data/ --unzip -q'
            # download the dataset file
            output, succes = syscall(download_cmd)
            if not succes:
                print('Error while downloading dataset:', dataset_name)
                print(output)

        # List the notebooks
        notebooks = []
        for i in range(1, 2):
            list_cmd = f'kaggle kernels list --dataset {dataset_name} --page {i} ' \
                       '--page-size 30 --sort-by voteCount --language python --csv'
            output, succes = syscall(list_cmd)
            if output == 'No kernels found':
                break
            df = pd.read_csv(io.StringIO(output))
            notebooks.append(df)

        notebooks = pd.concat(notebooks, ignore_index=True).drop_duplicates('ref')

        # download 20 kernels
        count = 0
        for _, nb_row in tqdm(notebooks.iterrows(), total=len(notebooks)):
            nb_name = nb_row['ref']
            dir_path = nb_name.replace("/", "-")
            file_name = nb_name.split('/')[1]

            nb_download_cmd = f'kaggle kernels pull {nb_name} --path ../CFGDemo/{dataset_path}/notebooks/{dir_path} --metadata'
            output, succes = syscall(nb_download_cmd)
            if not succes:
                print('Error while downloading kernel:', nb_name, '; Dataset:', dataset_name)
                continue
            count += 1
            if os.path.isfile(f'../CFGDemo/{dataset_path}/notebooks/{dir_path}/{file_name}.ipynb'):
                to_python_file_cmd = f'jupyter nbconvert ../CFGDemo/{dataset_path}/notebooks/{dir_path}/{file_name}.ipynb ' \
                                     f'--to script'
                output, success = syscall(to_python_file_cmd)
            
                if not success:
                    print('Error while tranforming file:', nb_name, '; Dataset:', dataset_name)
                else:
                    os.remove(f'../CFGDemo/{dataset_path}/notebooks/{dir_path}/{file_name}.ipynb')

            if os.path.isdir(f'../CFGDemo/{dataset_path}/notebooks/{dir_path}'):
                elements = os.listdir(f'../CFGDemo/{dataset_path}/notebooks/{dir_path}')
                if f'{file_name}.py' in elements:
                    with open(f'../CFGDemo/{dataset_path}/notebooks/{dir_path}/pipeline_info.json', 'w') as f:
                        json.dump({
                            'url': f"https://www.kaggle.com/{nb_row['ref']}",
                            'title': nb_row['title'],
                            'author': nb_row['author'],
                            'votes': nb_row['totalVotes'],
                            'score': random.uniform(0.5, 1.0),
                            'date': nb_row['lastRunTime']
                        }, f)

            if count >= 20:
                break


if __name__ == '__main__':
    main()
