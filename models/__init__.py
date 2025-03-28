# from .vavanilla_resnet_cifar import vanilla_resnet20
from .vanilla_models.vanilla_resnet_cifar import vanilla_resnet20
from .vanilla_models.vanilla_resnet_imagenet import resnet18


from .quan_resnet_imagenet import resnet18_quan, resnet34_quan

from .quan_alexnet_imagenet import alexnet_quan

from .quan_mobilenet_imagenet import mobilenet_v2_quan
from .vanilla_models.vanilla_mobilenet_imagenet import mobilenet_v2


from .quan_vgg import vgg11_quan, vgg13_quan, vgg16_quan, vgg19_quan
from .quan_hardened_vgg import vgg11_quan_hardened, vgg13_quan_hardened, vgg16_quan_hardened, vgg19_quan_hardened

from .quan_resnet import cifar100_resnet20, cifar100_resnet32, cifar100_resnet44, cifar100_resnet56
from .quan_hardened_resnet import cifar100_resnet20_hardened, cifar100_resnet32_hardened, cifar100_resnet44_hardened, cifar100_resnet56_hardened

from .quan_mobilenet_cifar10 import mobilenet_quan_cifar10
from .quan_mobilenet_cifar100 import mobilenet_quan_cifar100
