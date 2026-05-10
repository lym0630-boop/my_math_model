"""
为没有写权限的模型目录创建一个修复版副本：
- tokenizer_config.json 中 extra_special_tokens list -> {}
- 其余所有文件用软链接指向原目录，不占用额外空间
用法:
    python3 fix_tokenizer.py <原模型路径> <新目录路径>
"""
import os
import sys
import json
import shutil

def fix_model_dir(src_dir, dst_dir):
    src_dir = os.path.abspath(src_dir)
    dst_dir = os.path.abspath(dst_dir)

    os.makedirs(dst_dir, exist_ok=True)

    fixed = 0
    linked = 0

    for fname in os.listdir(src_dir):
        src_file = os.path.join(src_dir, fname)
        dst_file = os.path.join(dst_dir, fname)

        if os.path.exists(dst_file) or os.path.islink(dst_file):
            os.remove(dst_file)

        # tokenizer_config.json 单独处理
        if fname == "tokenizer_config.json":
            with open(src_file, "r") as f:
                tc = json.load(f)
            if isinstance(tc.get("extra_special_tokens"), list):
                print(f"  修复 extra_special_tokens: list -> {{}}")
                tc["extra_special_tokens"] = {}
                fixed += 1
            with open(dst_file, "w") as f:
                json.dump(tc, f, ensure_ascii=False, indent=2)
        else:
            # 其他文件直接软链接
            os.symlink(src_file, dst_file)
            linked += 1

    print(f"完成！修复文件: {fixed} 个，软链接: {linked} 个")
    print(f"新模型目录: {dst_dir}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python3 fix_tokenizer.py <原模型路径> <新目录路径>")
        sys.exit(1)
    fix_model_dir(sys.argv[1], sys.argv[2])
