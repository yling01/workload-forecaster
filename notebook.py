# terminal
"""
git clone https://github.com/yinfredyue/785-project.git
"""

# cell
"""
!pip install wandb
"""

# cell
"""
from google.colab import drive
drive.mount('/content/drive')
"""

# cell
"""
! cd /content/785-project/data && python preprocess.py
"""

# cell
"""
! python /content/785-project/src/deepar/main.py \
--data_dir /content/785-project/data/1year \
--data_name 1year \
--colab true \
--ckpt_dir /content/drive/MyDrive/785_project/checkpoints/ \
--plot_dir /content/drive/MyDrive/785_project/plots/
"""
