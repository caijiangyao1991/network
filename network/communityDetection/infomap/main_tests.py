import sys
from infomap_tests import Infomap_tests

def main():
    tests = Infomap_tests()
    out_file = open("tests_results.txt", 'w')
    out_str = tests.run_tests()
    if not out_str:
        out_file.write("Tests were successful")
    else:
        out_file.write(out_str)
    out_file.close()

if __name__ == "__main__":
    main()
    sys.exit()
