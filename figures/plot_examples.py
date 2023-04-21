import random
import pickle

import matplotlib.pyplot as plt
from torchvision import datasets

# mnist = datasets.MNIST('../data', train=True, download=True)
# plt.figure(figsize=(10,2))
# for i in range(5):
#     plt.subplot(1,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(mnist[i][0], cmap=plt.cm.binary)
#     plt.xlabel(mnist[i][1], fontsize=20, fontweight='bold')
#
# plt.tight_layout()
# plt.savefig("mnist_dataset.png")

NUM_CLASSES = 100
TRAIN_IMAGES_PER_CLASS = 50
TEST_IMAGES_PER_CLASS = 10

import pickle
from torch.utils.data import Subset
from torchvision import datasets, transforms
from transformers import ViTForImageClassification, ViTImageProcessor

from torch import nn
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
model.classifier = nn.Identity()

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

def my_normalization(x):
    return model(processor(x, return_tensors='pt')['pixel_values']).logits.squeeze()


transform = transforms.Compose([transforms.Lambda(my_normalization)])

inat_train = Subset(datasets.INaturalist('../data', version='2021_train_mini', download=False, transform=transform), [0])
# inat_test = datasets.INaturalist('../data', version='2021_valid', download=False)

print(inat_train[0][0].shape)
exit()
# random.seed(0)
# sampled_classes = random.sample(range(len(inat_train.all_categories)), NUM_CLASSES)
#
# train_indices = [TRAIN_IMAGES_PER_CLASS*cls + offset
#                  for cls in sampled_classes
#                  for offset in range(TRAIN_IMAGES_PER_CLASS)]
# test_indices = [TEST_IMAGES_PER_CLASS*cls + offset
#                 for cls in sampled_classes
#                 for offset in range(TEST_IMAGES_PER_CLASS)]
#
# with open("inat_sampled_classes", "wb") as f:
#     pickle.dump(sampled_classes, f)
#
# with open("inat_train_indices", "wb") as f:
#     pickle.dump(train_indices, f)
#
# with open("inat_test_indices", "wb") as f:
#     pickle.dump(test_indices, f)

with open("inat_sampled_classes", "rb") as f:
    sampled_classes = pickle.load(f)

# with open("inat_train_indices", "rb") as f:
#     train_indices = pickle.load(f)
#
# with open("inat_test_indices", "rb") as f:
#     test_indices = pickle.load(f)
#
def decode_inat(id: int) -> str:
    return " ".join(inat_test.category_name("full", id).split('_')[6:])

plt.figure(figsize=(10, 6))
random.seed(0)
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(inat_test[10*sampled_classes[i]][0], cmap=plt.cm.binary)
    plt.xlabel(decode_inat(inat_test[10*sampled_classes[i]][1]), fontsize=11, fontweight='bold')

plt.tight_layout()
# plt.show()
plt.savefig("inat_dataset.png")