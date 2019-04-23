import argparse
from flower_classifier import FlowerClassifier
import util


parser = argparse.ArgumentParser()
parser.add_argument("checkpoint_path", default="./checkpoint.pth", help="where checkpoint lives")
parser.add_argument("--gpu", default=False, action="store_true", help="use gpu or not")
parser.add_argument("--arch", default="vgg11", help="which pre-trained model to use as a base. vgg11 or resnet18")
parser.add_argument("--hidden_units", type=int, default=1024, help="size of hidden layer")
parser.add_argument("--top_k", type=int, default=5, help="number of top probabilities returned")
parser.add_argument("--category_names", default="cat_to_name.json", help="category to name mappings")
args = parser.parse_args()
print(args)

def main():
    f_class = FlowerClassifier(args.arch, args.hidden_units, args.gpu)
    f_class = util.load_checkpoint(f_class, args.checkpoint_path)
    top_probs, top_classes = f_class.predict('flowers/valid/1/image_06765.jpg', args.top_k, args.category_names)
    print(top_probs, top_classes)

if __name__ == "__main__": main()