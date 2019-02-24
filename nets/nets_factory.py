from nets import alexnet_224
from nets import alexnet_cifar

nets_map = {
    'alexnet_224' : alexnet_224,
    'alexnet_cifar' : alexnet_cifar,
}

def get_network(name):
    if name not in nets_map:
        raise ValueError('Name of net unkonw %s' % name)
    return nets_map[name]