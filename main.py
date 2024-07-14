import subprocess

commands = [
    'pip install -r ./Language-Model/requirement.txt',
    'python ./Language-Model/fastText.py',
    'python ./Language-Model/make_data.py',
    'python ./Language-Model/svd.py',
    'python ./Language-Model/word2vec.py',
    'python ./Language-Model/using_model_demo.py',
    'python ./Language-Model/visualize.py'
]

for command in commands:
    print(f"Executing command: {command}")
    process = subprocess.Popen(command, shell=True)
    process.wait() 

print("All commands executed successfully.")