# (c) Alex Spirin 2022-2023

import subprocess
import requests
import os 
import sys
import traceback
from IPython.utils import io

def pipi(modulestr):
  res = subprocess.run(['python','-m','pip', '-q', 'install', modulestr], stdout=subprocess.PIPE).stdout.decode('utf-8')
  print(res)

def replace_line_at_number(file_path, line_number, new_line):
    # Read the original file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Replace the desired line
    if 0 < line_number <= len(lines):
        lines[line_number - 1] = new_line + '\n'  # Add newline character

    # Write the modified content back to the file
    with open(file_path, 'w') as file:
        file.writelines(lines)

def replace_line(file_path, old_line, new_line):
    # Read the original file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    for i in range(len(lines)):
      lines[i] = lines[i].replace(old_line, new_line)

    # Write the modified content back to the file
    with open(file_path, 'w') as file:
        file.writelines(lines)

def gitclone(url, recursive=False, dest=None, branch=None):
  command = ['git', 'clone']
  if branch is not None:
    command.append(['-b', branch])
  command.append(url)
  if dest: command.append(dest)
  if recursive: command.append('--recursive')

  res = subprocess.run(command, stdout=subprocess.PIPE).stdout.decode('utf-8')
  print(res)

def nukedir(dir):
    if dir[-1] == os.sep: dir = dir[:-1]
    files = os.listdir(dir)
    for file in files:
        if file == '.' or file == '..': continue
        path = dir + os.sep + file
        if os.path.isdir(path):
            nukedir(path)
        else:
            os.unlink(path)
    os.rmdir(dir)

def gitpull(dir, force_branch=None, reset=False, commit=None, target_repo=None):
  cwd = os.getcwd()
  try:
      if not os.path.exists(dir) and target_repo is not None:
        gitclone(target_repo)
        return
      if os.path.exists(dir):
        print(f"pulling a fresh {dir.split('/')[-1]}")
        os.chdir(dir)
        if target_repo is not None:
          res = subprocess.run(['git', 'config', '--get', 'remote.origin.url'], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
          res = res.stdout.decode()
          if str(res).strip() != str(target_repo).strip():
            subprocess.run(['git', 'remote', 'remove', 'origin'])
            print(f'Current repo pointing to {res} instead of {target_repo}. Deleting and cloning from the right one.')
            print(f'If you get an access denied error here, please manually delete the folder {dir} and re-run this cell.')
            os.chdir(cwd)
            nukedir(dir)
            gitclone(target_repo)
            return

        res = subprocess.run(['git', 'stash'])
        if force_branch:
          res = subprocess.run(['git','rev-parse' '--verify' f'origin/{force_branch}']).returncode
          if res == 0:
            print(subprocess.run(['git', 'switch', '-C', force_branch], stderr=subprocess.PIPE).stderr.decode('utf-8'))
            print(subprocess.run(['git','branch',f'--set-upstream-to=origin/{force_branch}',force_branch], stderr=subprocess.PIPE).stderr.decode('utf-8'))
        if commit is not None:
          res = subprocess.run(['git', 'pull','origin',commit], stderr=subprocess.PIPE)
        else:
          res = subprocess.run(['git', 'pull'], stderr=subprocess.PIPE)
        print(res.stderr.decode())
        os.chdir(cwd)
  except:
    print(traceback.format_exc())
    pass

def pull_repos(is_colab):
  gitpull('./ComfyUI', reset=True, target_repo='https://github.com/Sxela/ComfyUI')

  file_path = './ComfyUI/comfy/sd.py'  # Replace with your file path
  new_line_content = "#from . import clip_vision"
  old_line = "from . import clip_vision"
  replace_line(file_path, old_line, new_line_content)

  gitpull('./stablediffusion')
  gitpull('./ControlNet')
  gitpull('./k-diffusion')
  gitpull('./WarpFusion')
  return 0

def get_version(package):
  proc = subprocess.run(['pip','show', package], stdout=subprocess.PIPE)
  out = proc.stdout.decode('UTF-8')
  returncode = proc.returncode
  if returncode != 0:
    return -1
  return out.split('Version:')[-1].split('\n')[0]

def uninstall_pytorch(is_colab):
  print('Uninstalling torch...')
  subprocess.run(['pip','uninstall','torch','-y'])
  subprocess.run(['pip','uninstall','torchvision','-y'])
  subprocess.run(['pip','uninstall','torchaudio','-y'])
  subprocess.run(['pip','uninstall','cudatoolkit','-y'])
  subprocess.run(['pip','uninstall','torchtext','-y'])
  subprocess.run(['pip','uninstall','xformers','-y'])
  if not is_colab:
    subprocess.run(['conda','uninstall','pytorch', 'torchvision',  
    'torchaudio',  'cudatoolkit', 'xformers','-y'])
  return 0


def install_dependencies_colab(is_colab, root_dir):
  """
  timm version can't be upgraded because it will make zoe depth checkpoint fail to load due to an updated layer naming convention
  """
  subprocess.run(['python','-m','pip','-q','install','tqdm','ipywidgets==7.7.1','protobuf==3.20.3'])
  from tqdm.notebook import tqdm
  progress_bar = tqdm(total=51)
  progress_bar.set_description("Installing dependencies")
  with io.capture_output(stderr=False) as captured:
    subprocess.run(['python','-m','pip','-q','install','mediapipe','piexif'])
    subprocess.run(['python','-m','pip','-q','install','safetensors==0.3.2','lark'])
    subprocess.run(['python','-m','pip','-q','uninstall','torchtext','-y'])
    progress_bar.update(3) #10
    gitclone('https://github.com/Sxela/sxela-stablediffusion', dest = 'stablediffusion')
    gitclone('https://github.com/Sxela/ControlNet-v1-1-nightly', dest = 'ControlNet')
    gitclone('https://github.com/pengbo-learn/python-color-transfer')
    gitclone('https://github.com/Sxela/generative-models')
    gitclone('https://github.com/Sxela/ComfyUI')

    progress_bar.update(3) #20
    try:
      if os.path.exists('./stablediffusion'):
        print('pulling a fresh stablediffusion')
        os.chdir( f'./stablediffusion')
        subprocess.run(['git', 'pull'])
        os.chdir( f'../')
    except:
      pass
    try:
        if os.path.exists('./ControlNet'):
          print('pulling a fresh ControlNet')
          os.chdir( f'./ControlNet')
          subprocess.run(['git', 'pull'])
          os.chdir( f'../')
    except: pass
    progress_bar.update(2) #25

    subprocess.run(['python','-m','pip','-q','install','-e','./stablediffusion'])
    progress_bar.update(2)
    pipi('ipywidgets==7.7.1')
    pipi('transformers==4.19.2')
    progress_bar.update(2)
    pipi('omegaconf')
    pipi('einops')
    pipi("pytorch_lightning>1.4.1,<=1.7.7")
    pipi('scikit-image')
    pipi('opencv-python')
    progress_bar.update(3) #30
    pipi('scikit-image')
    pipi('opencv-python')
    progress_bar.update(2)
    pipi('ai-tools')
    pipi('cognitive-face')
    progress_bar.update(2)
    pipi('zprint')
    pipi('kornia==0.5.0')

    progress_bar.update(2) #40
    subprocess.run(['python','-m','pip','-q','install','-e','git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers'])
    subprocess.run(['python','-m','pip','-q','install','-e','git+https://github.com/openai/CLIP.git@main#egg=clip'])
    progress_bar.update(2)
    pipi('lpips')
    progress_bar.update(2) #50
    gitclone('https://github.com/Sxela/k-diffusion')
    os.chdir( f'./k-diffusion')
    subprocess.run(['git', 'pull'])
    subprocess.run(['python','-m','pip','-q','install','-e','.'])
    os.chdir( f'../')
    sys.path.append('./k-diffusion')
    progress_bar.update(1) #60
    pipi('wget')
    pipi('webdataset')
    progress_bar.update(2)
    pipi('open_clip_torch')
    pipi('opencv-contrib-python==4.5.5.64')
    progress_bar.update(2)
    subprocess.run(['python','-m','pip','-q','uninstall','torchtext','-y'])
    subprocess.run(['python','-m','pip','-q','install','pandas','matplotlib'])
    progress_bar.update(2)
    pipi('fvcore')
    subprocess.run(['python','-m','pip','-q','install','datetime','ftfy','timm==0.6.13'])
    progress_bar.update(5)
    if is_colab:
      subprocess.run(['apt', 'install', 'imagemagick'], stdout=subprocess.PIPE).stdout.decode('utf-8')
    progress_bar.update(5)
    subprocess.run(['pip','install','onnxruntime-gpu','gdown'])

    try:
      from resize_right import resize
    except:
      if not os.path.exists("ResizeRight"):
        gitclone("https://github.com/assafshocher/ResizeRight.git")
      sys.path.append(f'{root_dir}/ResizeRight')
    progress_bar.update(1)
    if not os.path.exists("BLIP"):
        gitclone("https://github.com/salesforce/BLIP")
        sys.path.append(f'{root_dir}/BLIP')
    progress_bar.update(1) #75
    pipi('prettytable')
    pipi('fairscale')
    progress_bar.update(3) #80
    os.chdir(root_dir)
    subprocess.run(['git','clone','https://github.com/xinntao/Real-ESRGAN'])
    os.chdir('./Real-ESRGAN')
    pipi('basicsr')
    pipi('google-cloud-vision')
    pipi('ffmpeg')
    progress_bar.update(3) #9085
    subprocess.run(['python','-m','pip','-q','install','-r','requirements.txt'])
    progress_bar.update(1) #90
    subprocess.run(['python','setup.py','develop','-q'])
    os.chdir(root_dir)
    subprocess.run(['python','-m','pip','-q','install','torchmetrics==0.11.4'])

    file_path = f'{root_dir}/ComfyUI/comfy/sd.py'  # Replace with your file path
    line_number_to_replace = 14  # Replace with the line number you want to replace
    new_line_content = "#from . import clip_vision"
    old_line = "from . import clip_vision"

    replace_line(file_path, old_line, new_line_content)

def install_torch_colab(force_torch_reinstall, use_torch_v2):
      subprocess.run(['python', '-m', 'pip', '-q', 'install',
      'https://download.pytorch.org/whl/cu118/xformers-0.0.22.post4%2Bcu118-cp310-cp310-manylinux2014_x86_64.whl'])
      return 0

def install_torch_windows(force_torch_reinstall, use_torch_v2):
  torch_v2_install_failed = False
  if not os.path.exists('ffmpeg.exe'):
    url = 'https://github.com/GyanD/codexffmpeg/releases/download/6.0/ffmpeg-6.0-full_build.zip'
    print('ffmpeg.exe not found, downloading...')
    r = requests.get(url, allow_redirects=True)
    print('downloaded, extracting')
    open('ffmpeg-6.0-full_build.zip', 'wb').write(r.content)
    import zipfile
    with zipfile.ZipFile('ffmpeg-6.0-full_build.zip', 'r') as zip_ref:
        zip_ref.extractall('./')
    from shutil import copy
    copy('./ffmpeg-6.0-full_build/bin/ffmpeg.exe', './')
  torchver = get_version('torch')
  if torchver == -1: print('Torch not found.')
  else: print('Found torch:', torchver)
  if use_torch_v2:
    if torchver == -1 or force_torch_reinstall:
      print('Installing torch v2.')
      subprocess.run(['python', '-m', 'pip', '-q', 'install', 'torch==2.0.0', 
      'torchvision==0.15.1', '--upgrade', '--index-url', 'https://download.pytorch.org/whl/cu117', 'xformers'])
      try:
        import torch
        torch_v2_install_failed = not torch.cuda.is_available()
      except:
        torch_v2_install_failed = True
      if torch_v2_install_failed:
        print('Failed installing torch v2.')
      else:
        print('Successfully installed torch v2.')

  if not use_torch_v2:
    try:
      #check if we have an xformers installation
      import xformers
    except:
      if "3.10" in sys.version:
        if torchver == -1 or force_torch_reinstall:
            print('Installing torch v1.12.1')
            subprocess.run(['python', '-m','pip','-q','install','torch==1.12.1',
            'torchvision==0.13.1','--extra-index-url','https://download.pytorch.org/whl/cu113'])
        if "1.12" in get_version('torch'):
          print('Trying to install local xformers on Windows. Works only with pytorch 1.12.* and python 3.10.')
          subprocess.run(['python', '-m', 'pip', '-q', 'install',
            'https://github.com/C43H66N12O12S2/stable-diffusion-webui/releases/download/f/xformers-0.0.14.dev0-cp310-cp310-win_amd64.whl'])
        elif "1.13" in get_version('torch'):
          print('Trying to install local xformers on Windows. Works only with pytorch 1.13.* and python 3.10.')
          subprocess.run(['python' ,'-m', 'pip', '-q', 'install', 
           'https://github.com/C43H66N12O12S2/stable-diffusion-webui/releases/download/torch13/xformers-0.0.14.dev0-cp310-cp310-win_amd64.whl'])
  return 0
