import os
import zipfile

# 定义路径
base_dir = "/LSEM/user/chenyinda/code/signal_dy_static"
sub_dir = os.path.join(base_dir, "1104")
output_zip = os.path.join(sub_dir, "selected_files.zip")

def zip_selected_files(sub_dir, base_dir, output_zip):
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # 压缩 1104 目录下的所有 .py 和 .yaml 文件
        for item in os.listdir(sub_dir):
            full_path = os.path.join(sub_dir, item)
            if os.path.isfile(full_path) and (item.endswith(".py") or item.endswith(".yaml")):
                # 在压缩包中保留文件相对路径（只包含文件名）
                zipf.write(full_path, arcname=f"1104/{item}")

        # 压缩 signal_dy_static 目录下的所有 .md 文件
        for item in os.listdir(base_dir):
            full_path = os.path.join(base_dir, item)
            if os.path.isfile(full_path) and item.endswith(".md"):
                zipf.write(full_path, arcname=item)

    print(f"压缩完成: {output_zip}")

if __name__ == "__main__":
    zip_selected_files(sub_dir, base_dir, output_zip)