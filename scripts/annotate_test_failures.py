"""Report failed test identifiers as GitHub Actions annotations."""

import sys
from pathlib import Path
from xml.etree import ElementTree


def annotate(path):
    if not path.is_file():
        print("::error::pytest did not produce a test report")
        return
    for case in ElementTree.parse(path).iter("testcase"):
        if case.find("failure") is not None or case.find("error") is not None:
            print(f"::error::Failed test: {case.get('classname')}.{case.get('name')}")


if __name__ == "__main__":
    annotate(Path(sys.argv[1]))
