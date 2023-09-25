import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', default='./check_points')
parser.add_argument('-cn', '--code_name', )
parser.add_argument('-b', '--bucket',default='dutalota-eu' )
args = parser.parse_args()
print(args)

para = {
    "name": f"{args.code_name}.rar",
}

os.system(f'rm -rf ./{args.code_name}.rar')

last_file=sorted(os.listdir(f'{args.path}'),key=lambda x:int(x))[-1]
last_file_path=f'{args.path}/{last_file}'
print(last_file_path)

os.system(f'rar a ./{args.code_name}.rar {last_file_path}')

os.system(f'gsutil cp  {args.code_name}.rar  gs://{args.bucket}/{args.code_name}.rar')
