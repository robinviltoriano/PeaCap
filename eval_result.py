import os
import argparse
from evaluation.pycocoevalcap.eval import COCOEvalCap
def get_score(result_file):
    # Manually set JAVA_HOME for Python
    os.environ['JAVA_HOME'] = '/home/robin/java/jdk-24'
    os.environ['PATH'] = os.environ['JAVA_HOME'] + '/bin:' + os.environ['PATH']
    
    cocoEval = COCOEvalCap(result_file)
    cocoEval.evaluate()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-rf","--result_file", type=str, required=True, help="Path to the result JSON file")
    args = parser.parse_args()
    get_score(args.result_file)
