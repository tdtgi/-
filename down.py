#模型下载
import os
# from huggingface_hub import snapshot_download
from modelscope import snapshot_download
os.environ ['MODELSCOPE_CACHE']='.'
model_dir = snapshot_download('Shanghai_AI_Laboratory/internlm-xcomposer2-4khd-7b')
