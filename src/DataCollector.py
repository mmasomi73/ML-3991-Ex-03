import os

from colorama import Fore

from HodaDatasetReader import read_hoda_dataset


class DataCollector:
    train_dataset_path = '../dataset/Train 60000.cdb'
    test_dataset_path = '../dataset/Test 20000.cdb'
    naming_dataset_path = '../dataset/RemainingSamples.cdb'

    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    X_naming = []
    Y_naming = []

    log = False

    def __init__(self, log=False):
        self.log = log
        if log:
            print(Fore.GREEN + "\t+----------------------------------------------")
            print(Fore.GREEN + "\tRead Dataset Begin.")

        # self.X_train, self.Y_train = self.dataSetReader(self.train_dataset_path)
        self.X_test, self.Y_test = self.dataSetReader(self.test_dataset_path)
        # self.X_naming, self.Y_naming = self.dataSetReader(self.naming_dataset_path)

        if log:
            print(Fore.BLUE + "\tExists {} record in Train Data.".format(len(self.X_train)))
            print(Fore.BLUE + "\tExists {} record in Test Data.".format(len(self.X_test)))
            print(Fore.BLUE + "\tExists {} record in Renaming Data.".format(len(self.X_naming)))
            print(Fore.GREEN + "\t+----------------------------------------------")

    def dataSetReader(self, path):

        if not os.path.exists(path):
            print(Fore.RED + "\t+------------------------+")
            print(Fore.RED + "\t| File or Path Incorrect |")
            print(Fore.RED + "\t+------------------------+")

        return read_hoda_dataset(dataset_path=path,
                                 images_height=32,
                                 images_width=32,
                                 one_hot=False,
                                 reshape=True)

    def getXTrain(self):
        return self.X_train

    def getYTrain(self):
        return self.Y_train

    def getXTest(self):
        return self.X_test

    def getYTest(self):
        return self.Y_test

    def getXNaming(self):
        return self.X_naming

    def getYNaming(self):
        return self.Y_naming
