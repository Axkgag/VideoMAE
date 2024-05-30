from vecpredictor import Vecpredictor
import argparse


parser=argparse.ArgumentParser()
parser.add_argument('-i', 
                    '--img_folder', 
                    default='./TODO/retrieval/task1/gallery', 
                    help='data used to build gallery'
                )

parser.add_argument('-m',
                    '--model_dir_path',
                    default='./VEC_MODEL/task1/videomae_ssv2',
                    help='model_dir'
                )


if __name__=="__main__":
    args=parser.parse_args()

    vecpred = Vecpredictor()

    if vecpred.loadModel(args.model_dir_path, False):
        vecpred.genRetrievalBank(args.img_folder, part_info=[644, 446, 1633, 1424])
        # vecpred.genRetrievalBank(args.img_folder)
