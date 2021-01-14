import processing as pr
import knnBuild as knn
import svmBuild as svm
import annBuild as ann
import cnnBuild as cnn


def run():
    """
    This method will run the complete program for all algorithms and variations. This includes downloading
    and extracting the dataset from the web and processing it.
    :return:
    """
    pr.run()
    knn.run()
    svm.run()
    ann.run()
    cnn.run()


if __name__ == '__main__':
    run()
