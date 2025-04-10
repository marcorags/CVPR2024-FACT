import os

def get_project_base():
    src_dir = os.path.dirname(os.path.realpath(__file__))
    base = os.path.join(os.path.dirname(src_dir), "CVPR2024-FACT")
    return base


if __name__ == "__main__":
    print(get_project_base())
