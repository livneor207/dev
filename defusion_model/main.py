"""Dogs vs Cats diffusion model.

Train on https://www.kaggle.com/c/dogs-vs-cats

    python train.py --data-dir ./data --epochs 50

Generate cats and dogs from the best checkpoint:

    python sample.py --checkpoint ./outputs/model.pth --class-name both
"""

from train import main


if __name__ == '__main__':
    main()
