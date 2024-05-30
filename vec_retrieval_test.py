from vecpredictor import Vecpredictor
import argparse
import os
import time
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser()
parser.add_argument('-t', 
                    '--test_dir', 
                    default='./TODO/retrieval/task1/test/', 
                    help='data used to build gallery'
                )

parser.add_argument('-m',
                    '--model_dir_path',
                    default='./VEC_MODEL/task1/videomae_ssv2',
                    help='model_dir'
                )


def get_image_list(img_file):
    imgs_lists = []
    if img_file is None or not os.path.exists(img_file):
        raise Exception("not found any img file in {}".format(img_file))

    # img_end = ['jpg', 'png', 'jpeg', 'JPEG', 'JPG', 'bmp']clear
    img_end = ['mp4']
    if os.path.isfile(img_file) and img_file.split('.')[-1] in img_end:
        imgs_lists.append(img_file)
    elif os.path.isdir(img_file):
        for single_file in os.listdir(img_file):
            if single_file.split('.')[-1] in img_end:
                imgs_lists.append(os.path.join(img_file, single_file))
    if len(imgs_lists) == 0:
        raise Exception("not found any img file in {}".format(img_file))
    imgs_lists = sorted(imgs_lists)
    return imgs_lists

if __name__=="__main__":
    args=parser.parse_args()

    vecpred = Vecpredictor()

    cnt = 0
    T = 0

    y_label = []
    y_pred = []

    CLASS_ = ['a' + str(i) for i in range(10)]

    error_list = []

    if vecpred.loadModel(args.model_dir_path):
        video_list = get_image_list(args.test_dir)
        for idx, video_file in enumerate(video_list):
            video_name = os.path.basename(video_file)
            t0 = time.time()
            output = vecpred.predict(video_file, [644, 446, 1633, 1424])
            # output = vecpred.predict(video_file)
            t1 = time.time()
            # print(output)
            results = True if video_name[-5] == output['obj_class'][-1] else False

            y_label.append(int(video_name[-5]))
            y_pred.append(int(output['obj_class'][-1]))

            if not results:
                error_list.append(output["cosine"])

            if results:
                cnt += 1

            T += (t1 - t0) * 1000
            print(video_name, output['obj_class'], results, output["cosine"], (t1 - t0) * 1000)
    
        print(cnt * 100 / len(video_list), T / len(video_list))
        print(error_list)
        
        cm = confusion_matrix(y_label, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_, yticklabels=CLASS_)

        plt.xlabel("Predict")
        plt.ylabel("GroundTruth")
        plt.title("ConfusionMatrix")
        plt.show()


