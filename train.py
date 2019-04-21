import argparse
from flower_classifier import FlowerClassifier

parser = argparse.ArgumentParser()
parser.add_argument("data_dir", help="path to training images")
parser.add_argument("--save_dir", help="path where checkpoint is saved")
parser.add_argument("--arch", default="vgg11", help="which pre-trained model to use as a base. vgg11 or resnet18")
parser.add_argument("--learning_rate", type=float, default=0.003, help="learning rate of the model")
parser.add_argument("--hidden_units", type=int, default=1024, help="size of hidden layer")
parser.add_argument("--gpu", default=False, action="store_true", help="size of hidden layer")
parser.add_argument("--epochs", type=int, default=1, help="number of training epochs")
args = parser.parse_args()
print(args)

def main():
    model = FlowerClassifier(args.arch, args.hidden_units, args.gpu)
    model.train(data_dir=args.data_dir, epochs=args.epochs, learning_rate=args.learning_rate)
    #print(model.cat_to_name)

if __name__ == "__main__": main()
