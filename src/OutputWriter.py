import os

import matplotlib.pyplot as plt


class OutputWriter:
    train_img_path = '../outs/train/'
    test_img_path = '../outs/test/'
    naming_img_path = '../outs/naming/'

    def __init__(self):
        print('Hello')
        # self.path = path
        # self.data = data
        # self.algos = algos

    #
    # def write(self):
    #     file_name = self.path + self.algos + '.csv'
    #     file = open(file_name, "w+")
    #     file.write(self.algos + ' Results:\n')
    #     file.write('False Alarm Rate,{}\n'.format(self.data['far']))
    #     file.write('Missing Alarm Rate,{}\n'.format(self.data['mar']))
    #     file.write('Accuracy Rate,{}\n'.format(self.data['acc']))
    #     file.write('Train Time,{}\n'.format(self.data['tr']))
    #     file.write('Test Time,{}\n'.format(self.data['te']))
    #     file.write('True Positive,{}\n'.format(self.data['tp']))
    #     file.write('True Negative,{}\n'.format(self.data['tn']))
    #     file.write('False Positive,{}\n'.format(self.data['fp']))
    #     file.write('False Negative,{}\n'.format(self.data['fn']))

    def imageWriter(self, image, label, imgType='train'):
        label = str(int(label))
        if imgType == 'train':
            base_path = self.train_img_path
        elif imgType == 'test':
            base_path = self.test_img_path
        elif imgType == 'naming':
            base_path = self.naming_img_path
        else:
            print("\tError in image data type")
            return

        base_path += str(label)
        if not os.path.exists(base_path):
            os.makedirs(base_path)
            print("\tPath {} was Created.".format(base_path))

        fig = plt.figure(figsize=(5, 5))
        counter = len([f for f in os.listdir(base_path + '/') if os.path.isfile(os.path.join(base_path + '/', f))]) + 1
        full_name = base_path + '/' + str(label) + '-' + str(counter) + '.png'
        plt.imshow(image.reshape([32, 32]), cmap='gray')
        plt.savefig(full_name, format='png')
        plt.close('all')
