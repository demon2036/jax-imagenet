import json
import os

import requests
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', default='./check_points')
parser.add_argument('-tk', '--token', )
parser.add_argument('-cn', '--code_name', )
args = parser.parse_args()
print(args)

headers = {
    "Authorization": f"Bearer {args.token}"
}

para = {
    "name": f"{args.code_name}.rar",
}

os.system(f'rm -rf ./{args.code_name}.rar')

last_file=sorted(os.listdir(f'{args.path}'),key=lambda x:int(x))[-1]
last_file_path=f'{args.path}/{last_file}'


os.system(f'rar a ./{args.code_name}.rar {last_file_path}')


files = {
    'data': ('metadata', json.dumps(para), 'application/json;charset=UTF-8'),#charset=UTF-8
    'file': open(f'./{args.code_name}.rar', 'rb')
}




r = requests.post("https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart",
                  headers=headers,
                  files=files
                  )

print(r.text)
