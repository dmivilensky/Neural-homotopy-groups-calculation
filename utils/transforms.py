import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy.random as rnd
import itertools as it


class ToTensor:
    def __init__(self, *args):
        self.to_args = list(args)

    def __call__(self, item):
        item = torch.tensor(item)
        for to_arg in self.to_args:
            item = item.to(to_arg)
        return item


class Pad:
    def __init__(self, length, mode='center'):
        self.length = length
        self.mode = mode

        if not self.mode in ['left', 'center', 'right', 'random', 'random-side']:
            raise ValueError

    def __call__(self, item):
        zeros = self.length - len(item)
        if self.mode == 'left':
            return Pad.__left__(item, zeros)
        if self.mode == 'center':
            return Pad.__center__(item, zeros)
        if self.mode == 'right':
            return Pad.__right__(item, zeros)
        if self.mode == 'random':
            return Pad.__random__(item, zeros)
        if self.mode == 'random-side':
            return Pad.__random_from_sides__(item, zeros)

    @staticmethod
    def __left__(item, zeros):
        return F.pad(item, (zeros, 0))

    @staticmethod
    def __center__(item, zeros):
        return F.pad(item, (zeros // 2, zeros - zeros // 2))

    @staticmethod
    def __right__(item, zeros):
        return F.pad(item, (0, zeros))

    @staticmethod
    def __random__(item, zeros):
        count_by_index = [0] * (item.size(0) + 1)
        for idx in torch.randint(0, item.size(0) + 1, (zeros, )):
            count_by_index[idx] += 1
        to_cat = []
        for zeros_count, element in zip(
            count_by_index, 
            it.chain(torch.split(item, 1), [None])
        ):
            if zeros_count > 0:
                to_cat.append(torch.zeros(zeros_count))
            if not element is None:
                to_cat.append(element) 
        return torch.cat(to_cat)

    @staticmethod
    def __random_from_sides__(item, zeros):
        left_zeros = rnd.randint(0, zeros + 1)
        right_zeros = zeros - left_zeros
        return F.pad(item, (left_zeros, right_zeros))


class MakeRowZero:
    def __init__(self, generators_number, row):
        tensor = torch.ones(2 * generators_number + 1)
        tensor[row] = 0
        self.tensor = tensor.detach().clone()

    def __call__(self, item):
        return item * self.tensor[:, None]



class OneHotEncoder:
    def __init__(self, generators_number):
        self.generators_number = generators_number

    def __call__(self, item):
        return F.one_hot(
                (item + self.generators_number).long(),
                num_classes = 2 * self.generators_number + 1
            ).transpose(0, 1)


class OneHotDecoder:
    def __init__(self, generators_number):
        self.tensor = torch.arange(start = -generators_number, end = generators_number + 1, step = 1)

    def __call__(self, item):
        return (item * self.tensor[:, None]).sum(dim = 0)
    
    
class CrossEncoder:
    def __init__(self, generators_number, with_pre_transform = True):
        self.pre_transform = None if not with_pre_transform else Compose([
                OneHotEncoder(generators_number),
                ToTensor(torch.float64)
            ])
        self.matrix = torch.cat([
            torch.flip(-torch.eye(generators_number), [1]),
            torch.zeros((generators_number, 1)),
            torch.eye(generators_number)
        ], dim = 1).to(torch.float64)

    def __call__(self, item):
        return self.matrix @ (item if self.pre_transform is None else self.pre_transform(item))


class FromOneHotEncoderToCrossEncoderDifferentiable(nn.Module):
    def __init__(self, generators_number):
        super().__init__()
        self.matrix = torch.cat([
            torch.flip(-torch.eye(generators_number), [1]),
            torch.zeros((1, generators_number)),
            torch.eye(generators_number)
        ], dim = 0).to(torch.float64)

    def to(*args, **kwargs):
        self.matrix.to(args, kwargs)
    
    def forward(self, batch):
        return batch.permute(0, 2, 1).matmul(self.matrix).permute(0, 2, 1)


class RandomNoise:
    def __init__(self, distr):
        self.distr = distr

    def __call__(self, item):
        noise = self.distr(item.size())
        return item + noise


def subgroup_by_label(label):
    result = []
    for idx in range(label.bit_length()):
        if label & (1 << (idx)):
            result.append(idx + 1)
    return result


import core.free_group as group

class FromSubgroupLabel:
    def __init__(self, label):
        self.subgroup = subgroup_by_label(label)

    def __call__(self, item):
        return group.is_from_normal_closure(self.subgroup, item)


class Softmax():
    def __init__(self):
        pass

    def __call__(self, item):
        return F.softmax(item, dim = 0)


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, v):
        for tr in self.transforms:
            v = tr(v)
        return v


class RandomChoice:
    def __init__(self, transforms, p=None):
        self.transforms = transforms
        self.p = p

    def __call__(self, v):
        transform = rnd.choice(self.transforms, size=1, p=self.p)[0]
        return transform(v)


class Identity:
    def __init__(self):
        pass

    def __call__(self, v):
        return v
