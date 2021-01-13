# must be run on windows. Automates creation of windows wheel
# might need this: https://docs.microsoft.com/en-us/answers/questions/136595/error-microsoft-visual-c-140-or-greater-is-require.html
import pathlib, sys, subprocess

if __name__ == "__main__":
    repo_folder = pathlib.Path(__file__).absolute().parents[0]
    c_build_folder = repo_folder / "src" / "py" / "c-build"
    try:
        c_build_folder.glob("*.pyd").__next__()
        print("Clear the c-build folder!")
        exit(1)
    except StopIteration:
        pass

    if subprocess.run(["pip", "install", repo_folder]).returncode != 0:
        print("compile failed")
        exit(1)

    packages_folder = None
    for i in sys.path:
        if "site-packages" in i:
            packages_folder = pathlib.Path(i)
            break
    if packages_folder is None:
        print("Couldn't find site-packages!")
        exit(1)
    pyd_file = (packages_folder / "sba" / "c-build").glob("*.pyd").__next__()
    
    if subprocess.run(["cmd", "/c", "copy", pyd_file, c_build_folder]).returncode != 0:
        print("Copy failed")
    
    if subprocess.run(["python3", "setup.py", "bdist_wheel"]).returncode != 0:
        print("wheel creation failed")
        exit(1)
    
    subprocess.run(["pip", "uninstall", "-y", "sparse-bit-array"])

    print("Installing wheel...")
    if subprocess.run(["pip", "install", (repo_folder / "dist").glob("*.whl").__next__()]).returncode != 0:
        print("wheel creation failed")
        exit(1)
    
    print("Testing wheel...")
    subprocess.run(["python3", (repo_folder / "tests" / "tests.py")])


